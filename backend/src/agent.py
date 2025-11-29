# IMPROVE THE AGENT AS PER YOUR NEED 1
"""
Day 8 – Voice Game Master (D&D-Style Adventure) - Voice-only GM agent

- Uses LiveKit agent plumbing similar to the provided food_agent_sqlite example.
- GM persona, universe, tone and rules are encoded in the agent instructions.
- Keeps STT/TTS/Turn detector/VAD integration untouched (murf, deepgram, silero, turn_detector).
- Tools:
    - start_adventure(): start a fresh session and introduce the scene
    - get_scene(): return the current scene description (GM text) ending with "What do you do?"
    - player_action(action_text): accept player's spoken action, update state, advance scene
    - show_journal(): list remembered facts, NPCs, named locations, choices
    - restart_adventure(): reset state and start over
- Userdata keeps continuity between turns: history, inventory, named NPCs/locations, choices, current_scene
"""

import json
import logging
import os
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("voice_game_master")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Simple Game World Definition
# -------------------------
# A compact world with a few scenes and choices forming a mini-arc.
WORLD = {
    "intro": {
        "title": "An F-Rank Gate Opens",
        "desc": (
            "You awaken on the cracked floor of a newly-appeared F-Rank Gate. "
            "Mana mist surrounds you, shimmering in hues of blue. A ruined outpost tower "
            "smolders a short distance inland, its barrier crystals shattered. A narrow path leads "
            "toward a cluster of abandoned hunter cottages to the east. "
            "Beside you in the dust lies a faintly glowing System Cube, half-buried."
        ),
        "choices": {
            "inspect_box": {
                "desc": "Examine the glowing System Cube.",
                "result_scene": "box",
            },
            "approach_tower": {
                "desc": "Walk toward the damaged hunter watchtower.",
                "result_scene": "tower",
            },
            "walk_to_cottages": {
                "desc": "Follow the path east toward the deserted cottages.",
                "result_scene": "cottages",
            },
        },
    },

    "box": {
        "title": "The System Cube",
        "desc": (
            "The cube hums softly with mana. When you touch it, a holographic map flashes into view—"
            "a dungeon layout with a marked symbol: 'Beneath the tower, the key resonates.' "
            "As you study it, the cracked tower emits a faint pulse, almost calling your name."
        ),
        "choices": {
            "take_map": {
                "desc": "Absorb the System map into your interface.",
                "result_scene": "tower_approach",
                "effects": {
                    "add_journal": "Obtained System Map: 'Beneath the tower, the key resonates.'"
                },
            },
            "leave_box": {
                "desc": "Leave the cube untouched.",
                "result_scene": "intro",
            },
        },
    },

    "tower": {
        "title": "Hunter Watchtower",
        "desc": (
            "The watchtower’s walls are cracked and glowing embers flicker inside. At its base lies "
            "an old mana-sealed hatch with an iron latch—ancient, but recently disturbed. "
            "You may try the latch blindly, search for another way in, or retreat."
        ),
        "choices": {
            "try_latch_without_map": {
                "desc": "Attempt to open the mana latch without any clue.",
                "result_scene": "latch_fail",
            },
            "search_around": {
                "desc": "Search the rubble for alternate dungeon entrances.",
                "result_scene": "secret_entrance",
            },
            "retreat": {
                "desc": "Return to the Gate’s entrance.",
                "result_scene": "intro",
            },
        },
    },

    "tower_approach": {
        "title": "Approaching the Tower",
        "desc": (
            "With the System Map guiding you, you approach the watchtower. The holographic markings align "
            "perfectly with the hatch. As you near it, you hear a faint mana resonance—almost like the latch is singing."
        ),
        "choices": {
            "open_hatch": {
                "desc": "Use the clue to carefully unlock the mana latch.",
                "result_scene": "latch_open",
                "effects": {
                    "add_journal": "Used System Map clue to unlock the dungeon hatch."
                },
            },
            "search_around": {
                "desc": "Search for other hidden mana entrances.",
                "result_scene": "secret_entrance",
            },
            "retreat": {
                "desc": "Return to the Gate’s starting point.",
                "result_scene": "intro",
            },
        },
    },

    "latch_fail": {
        "title": "Mana Backlash",
        "desc": (
            "You force the latch carelessly. The mana seal reacts violently—sending a tremor through the ground. "
            "Something large stirs inside the tower, awakened by your mistake."
        ),
        "choices": {
            "run_away": {
                "desc": "Retreat quickly back to the Gate entrance.",
                "result_scene": "intro",
            },
            "stand_ground": {
                "desc": "Stay and prepare to fight whatever emerges.",
                "result_scene": "tower_combat",
            },
        },
    },

    "latch_open": {
        "title": "Dungeon Access Granted",
        "desc": (
            "Following the System Map’s instructions, the latch opens with a click. A rush of cold mana escapes. "
            "A spiral staircase carved from stone descends into an underground dungeon chamber, lit by glowing mana moss."
        ),
        "choices": {
            "descend": {
                "desc": "Enter the dungeon’s lower chamber.",
                "result_scene": "cellar",
            },
            "close_hatch": {
                "desc": "Close the hatch and reconsider.",
                "result_scene": "tower_approach",
            },
        },
    },

    "secret_entrance": {
        "title": "Hidden Path",
        "desc": (
            "Behind collapsed rubble, you discover a narrow crawlspace. An old hunter rope leads downward—"
            "the air is thick with mana and the scent of iron. Something lurks deeper inside."
        ),
        "choices": {
            "squeeze_in": {
                "desc": "Enter the hidden tunnel.",
                "result_scene": "cellar",
            },
            "mark_and_return": {
                "desc": "Mark the tunnel for later.",
                "result_scene": "intro",
            },
        },
    },

    "cellar": {
        "title": "Mana Chamber",
        "desc": (
            "You enter a circular underground room. Runes glow faintly across the walls. "
            "In the center stands a stone pedestal holding a brass Dungeon Key and a sealed System Scroll."
        ),
        "choices": {
            "take_key": {
                "desc": "Take the Dungeon Key.",
                "result_scene": "cellar_key",
                "effects": {
                    "add_inventory": "dungeon_key",
                    "add_journal": "Obtained Dungeon Key from mana pedestal."
                },
            },
            "open_scroll": {
                "desc": "Break the System seal and read the scroll.",
                "result_scene": "scroll_reveal",
                "effects": {
                    "add_journal": "System Scroll: 'The water beast guards what hunters once lost.'"
                },
            },
            "leave_quietly": {
                "desc": "Leave the chamber and close the hatch.",
                "result_scene": "intro",
            },
        },
    },

    "cellar_key": {
        "title": "Key Resonance",
        "desc": (
            "As you hold the key, the runes dim. A hidden door opens, revealing a statue of an ancient S-Rank hunter. "
            "The statue glows and speaks: 'Will you return what was taken from this Gate?'"
        ),
        "choices": {
            "pledge_help": {
                "desc": "Pledge to restore what was lost.",
                "result_scene": "reward",
                "effects": {
                    "add_journal": "You pledged to resolve the Gate’s disturbance."
                },
            },
            "refuse": {
                "desc": "Pocket the key without answering.",
                "result_scene": "cursed_key",
                "effects": {
                    "add_journal": "You kept the key—its mana feels heavy and corrupted."
                },
            },
        },
    },

    "scroll_reveal": {
        "title": "System Scroll",
        "desc": (
            "The scroll reveals lore: a hunter heirloom was stolen by a water-type dungeon beast lurking beneath the tower. "
            "It hints that the Dungeon Key will react when used truthfully."
        ),
        "choices": {
            "search_for_key": {
                "desc": "Search the pedestal for the key.",
                "result_scene": "cellar_key",
            },
            "leave_quietly": {
                "desc": "Leave with the information.",
                "result_scene": "intro",
            },
        },
    },

    "tower_combat": {
        "title": "Dungeon Beast Emerges",
        "desc": (
            "A scaled, mana-soaked creature crawls from the tower’s shadows. Its eyes burn crimson. "
            "Its claws drip with corrupted water mana. It lunges at you."
        ),
        "choices": {
            "fight": {
                "desc": "Fight the dungeon beast.",
                "result_scene": "fight_win",
            },
            "flee": {
                "desc": "Retreat to the Gate’s entrance.",
                "result_scene": "intro",
            },
        },
    },

    "fight_win": {
        "title": "Victory",
        "desc": (
            "You defeat the creature. As it dissolves into mana particles, it drops a small engraved Hunter Locket—"
            "matching the relic described in the System Scroll."
        ),
        "choices": {
            "take_locket": {
                "desc": "Take the Hunter Locket.",
                "result_scene": "reward",
                "effects": {
                    "add_inventory": "hunter_locket",
                    "add_journal": "Recovered lost Hunter Locket."
                },
            },
            "leave_locket": {
                "desc": "Leave it behind and rest.",
                "result_scene": "intro",
            },
        },
    },

    "reward": {
        "title": "Gate Stabilized",
        "desc": (
            "A calm wave of mana spreads across the Gate. The distortion fades. "
            "A faint System message appears: 'Disturbance resolved. Mini-Arc Complete.' "
            "But the Gate remains vast—there are more secrets waiting inside."
        ),
        "choices": {
            "end_session": {
                "desc": "End here and return to the Gate entrance.",
                "result_scene": "intro",
            },
            "keep_exploring": {
                "desc": "Continue exploring the dungeon.",
                "result_scene": "intro",
            },
        },
    },

    "cursed_key": {
        "title": "Corrupted Key",
        "desc": (
            "The Dungeon Key pulses with a cold glow. A weight presses on your chest—"
            "a curse tied to an unfulfilled promise. The System warns: 'Penalty may occur.'"
        ),
        "choices": {
            "seek_redemption": {
                "desc": "Attempt to purify the key.",
                "result_scene": "reward",
            },
            "bury_key": {
                "desc": "Throw the key away and hope the curse fades.",
                "result_scene": "intro",
            },
        },
    },
}

# -------------------------
# Per-session Userdata
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    current_scene: str = "intro"
    history: List[Dict] = field(default_factory=list)  # list of {'scene', 'action', 'time', 'result_scene'}
    journal: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    named_npcs: Dict[str, str] = field(default_factory=dict)
    choices_made: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

# -------------------------
# Helper functions
# -------------------------
def scene_text(scene_key: str, userdata: Userdata) -> str:
    """
    Build the descriptive text for the current scene, and append choices as short hints.
    Always end with 'What do you do?' so the voice flow prompts player input.
    """
    scene = WORLD.get(scene_key)
    if not scene:
        return "You are in a featureless void. What do you do?"

    desc = f"{scene['desc']}\n\nChoices:\n"
    for cid, cmeta in scene.get("choices", {}).items():
        desc += f"- {cmeta['desc']} (say: {cid})\n"
    # GM MUST end with the action prompt
    desc += "\nWhat do you do?"
    return desc

def apply_effects(effects: dict, userdata: Userdata):
    if not effects:
        return
    if "add_journal" in effects:
        userdata.journal.append(effects["add_journal"])
    if "add_inventory" in effects:
        userdata.inventory.append(effects["add_inventory"])
    # Extendable for more effect keys

def summarize_scene_transition(old_scene: str, action_key: str, result_scene: str, userdata: Userdata) -> str:
    """Record the transition into history and return a short narrative the GM can use."""
    entry = {
        "from": old_scene,
        "action": action_key,
        "to": result_scene,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    userdata.history.append(entry)
    userdata.choices_made.append(action_key)
    return f"You chose '{action_key}'."

# -------------------------
# Agent Tools (function_tool)
# -------------------------

@function_tool
async def start_adventure(
    ctx: RunContext[Userdata],
    player_name: Annotated[Optional[str], Field(description="Player name", default=None)] = None,
) -> str:
    """Initialize a new adventure session for the player and return the opening description."""
    userdata = ctx.userdata
    if player_name:
        userdata.player_name = player_name
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"

    opening = (
        f"Greetings {userdata.player_name or 'traveler'}. Welcome to '{WORLD['intro']['title']}'.\n\n"
        + scene_text("intro", userdata)
    )
    # Ensure GM prompt present
    if not opening.endswith("What do you do?"):
        opening += "\nWhat do you do?"
    return opening

@function_tool
async def get_scene(
    ctx: RunContext[Userdata],
) -> str:
    """Return the current scene description (useful for 'remind me where I am')."""
    userdata = ctx.userdata
    scene_k = userdata.current_scene or "intro"
    txt = scene_text(scene_k, userdata)
    return txt

@function_tool
async def player_action(
    ctx: RunContext[Userdata],
    action: Annotated[str, Field(description="Player spoken action or the short action code (e.g., 'inspect_box' or 'take the box')")],
) -> str:
    """
    Accept player's action (natural language or action key), try to resolve it to a defined choice,
    update userdata, advance to the next scene and return the GM's next description (ending with 'What do you do?').
    """
    userdata = ctx.userdata
    current = userdata.current_scene or "intro"
    scene = WORLD.get(current)
    action_text = (action or "").strip()

    # Attempt 1: match exact action key (e.g., 'inspect_box')
    chosen_key = None
    if action_text.lower() in (scene.get("choices") or {}):
        chosen_key = action_text.lower()

    # Attempt 2: fuzzy match by checking if action_text contains the choice key or descriptive words
    if not chosen_key:
        # try to find a choice whose description words appear in action_text
        for cid, cmeta in (scene.get("choices") or {}).items():
            desc = cmeta.get("desc", "").lower()
            if cid in action_text.lower() or any(w in action_text.lower() for w in desc.split()[:4]):
                chosen_key = cid
                break

    # Attempt 3: fallback by simple keyword matching against choice descriptions
    if not chosen_key:
        for cid, cmeta in (scene.get("choices") or {}).items():
            for keyword in cmeta.get("desc", "").lower().split():
                if keyword and keyword in action_text.lower():
                    chosen_key = cid
                    break
            if chosen_key:
                break

    if not chosen_key:
        # If we still can't resolve, ask a clarifying GM response but keep it short and end with prompt.
        resp = (
            "I didn't quite catch that action for this situation. Try one of the listed choices or use a simple phrase like 'inspect the box' or 'go to the tower'.\n\n"
            + scene_text(current, userdata)
        )
        return resp

    # Apply the chosen choice
    choice_meta = scene["choices"].get(chosen_key)
    result_scene = choice_meta.get("result_scene", current)
    effects = choice_meta.get("effects", None)

    # Apply effects (inventory/journal, etc.)
    apply_effects(effects or {}, userdata)

    # Record transition
    _note = summarize_scene_transition(current, chosen_key, result_scene, userdata)

    # Update current scene
    userdata.current_scene = result_scene

    # Build narrative reply: echo a short confirmation, then describe next scene
    next_desc = scene_text(result_scene, userdata)

    # A small flourish so the GM sounds more persona-driven
    persona_pre = (
        "The Game Master (a calm, slightly mysterious narrator) replies:\n\n"
    )
    reply = f"{persona_pre}{_note}\n\n{next_desc}"
    # ensure final prompt present
    if not reply.endswith("What do you do?"):
        reply += "\nWhat do you do?"
    return reply

@function_tool
async def show_journal(
    ctx: RunContext[Userdata],
) -> str:
    userdata = ctx.userdata
    lines = []
    lines.append(f"Session: {userdata.session_id} | Started at: {userdata.started_at}")
    if userdata.player_name:
        lines.append(f"Player: {userdata.player_name}")
    if userdata.journal:
        lines.append("\nJournal entries:")
        for j in userdata.journal:
            lines.append(f"- {j}")
    else:
        lines.append("\nJournal is empty.")
    if userdata.inventory:
        lines.append("\nInventory:")
        for it in userdata.inventory:
            lines.append(f"- {it}")
    else:
        lines.append("\nNo items in inventory.")
    lines.append("\nRecent choices:")
    for h in userdata.history[-6:]:
        lines.append(f"- {h['time']} | from {h['from']} -> {h['to']} via {h['action']}")
    lines.append("\nWhat do you do?")
    return "\n".join(lines)

@function_tool
async def restart_adventure(
    ctx: RunContext[Userdata],
) -> str:
    """Reset the userdata and start again."""
    userdata = ctx.userdata
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"
    greeting = (
        "The world resets. A new tide laps at the shore. You stand once more at the beginning.\n\n"
        + scene_text("intro", userdata)
    )
    if not greeting.endswith("What do you do?"):
        greeting += "\nWhat do you do?"
    return greeting

# -------------------------
# The Agent (GameMasterAgent)
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        # System instructions define Universe, Tone, Role
        instructions = """
        You are 'sung jinwoo', the Game Master (GM) for a voice-only, Solo-Leveling–style dungeon adventure.
        
        Universe: A world of Gates, Dungeons, mana beasts, hunters, relics, and shifting mana anomalies.
                  The setting begins inside a newly formed low-rank Gate near an abandoned hunter outpost.
        
        Tone: Mysterious, dramatic, immersive, but not overly dark. Speak like a seasoned hunter guiding a rookie.
              Keep tension present, but avoid excessive horror. Prioritize clarity for voice gameplay.
        
        Role: You are the GM. You vividly describe dungeon scenes, mana flows, gates, enemies, loot, and System echoes.
              You must remember the player's past actions, inventory, relics, key items, and dungeon progression.
              Always guide the story and always end your message with: 'What do you do?'

        Rules:
            - Use the provided tools to start the adventure, get the current dungeon scene, accept the player's action,
              access the player's journal, or restart the dungeon run.
            - Maintain continuity using session userdata: reference the player's acquired relics, keys, maps,
              and any System messages they have uncovered.
            - Aim for short, meaningful turns just like a fast-paced Solo-Leveling arc.
            - Each GM message MUST end with 'What do you do?'.
            - Since this is voice-first, keep responses crisp, vivid, and easy to follow.
        """
        super().__init__(
            instructions=instructions,
            tools=[start_adventure, get_scene, player_action, show_journal, restart_adventure],
        )

# -------------------------
# Entrypoint & Prewarm (keeps speech functionality)
# -------------------------
def prewarm(proc: JobProcess):
    # load VAD model and stash on process userdata, try/catch like original file
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🎲" * 8)
    logger.info("🚀 STARTING VOICE GAME MASTER (Brinmere Mini-Arc)")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # Start the agent session with the GameMasterAgent
    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

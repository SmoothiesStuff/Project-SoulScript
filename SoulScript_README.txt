
########## SoulScript ##########
A LangGraph driven framework for giving NPCs the semblance of a soul
Version 0.1 preview

########## Vision ##########
SoulScript turns non player characters into individual agents with history, personality, memory, and evolving relationships. 
Each NPC carries a structured state. This includes backstory, stats, goals, motivations, traits, and both short term and long term memory. 
Developers define the character and the choice set. The system combines that with player or NPC inputs and uses a LangGraph flow to make decisions. 
Decisions lead to chosen actions and state updates. Memories get recorded and summarized so the character can refer back to what happened.
The result is a living town or party where relationships grow and matter. Every playthrough is unique.

Audience is indie developers building relationship heavy games. Think cozy life sims, town RPGs, party builders, and games where your story is shaped by who you spend time with.

########## Utility ##########
For players
* Unique playthroughs that feel personal
* Dynamic relationships that change based on what you do
* A world where NPCs talk to each other, not only to you

For developers
* Configurable NPC profiles with backstory, stats, motivations, personality, and secrets
* A memory system that records dialogue and events, then distills them into long term summaries
* Decision loops powered by LangGraph where you define actions like join party, fight, share item, gossip, teach skill, or say a line
* A live monitor where you can watch state and relationship graphs update in real time
* Planned Unity and Unreal bindings so you can drop SoulScript agents into your game

########## Core Concepts ##########
NPC state and context
* History and backstory set by the developer
* Stats that are numeric or categorical such as health, loyalty, courage, mood, curiosity
* Motivations and goals that guide choices
* Personality traits such as loyal, greedy, shy, brave, curious
* Secrets and dreams that may never be shown to the player but shape behavior
* Dynamic state that changes with interactions and time
* Memory with short term items and long term summaries

Decision making
* Choice types are defined by the developer such as join party, fight, talk, trade, share secret, ignore, help, comfort, accuse, apologize
* LangGraph nodes model the flow of decisions with edges that apply conditions and tools
* The LLM receives a grounded context window that includes NPC state, recent memories, salient long term summaries, and the current event
* The output is a chosen action and optional natural language like a dialogue line
* The system updates state, memories, relationships, and cooldowns, then returns to idle until the next input

Interactions
* NPC to player dialogue with storage and summarization
* NPC to NPC conversations so relationships can evolve without player input
* Relationship evolution with opinions that move in small bounded steps and can decay if neglected
* Event logging for later analysis or replay viewing

Developer tools early
* Text based profiles in JSON or YAML
* Simple LangGraph setup with a small library of reusable nodes
* Debug panel that shows NPC cards, a live relationship graph, and a rolling dialogue log

Demo environment
* A minimal town square where a handful of NPCs wander and talk
* Live visualization of opinions and relationship strength over time
* A reset button to re run the simulation and see how it changes

########## Flow from State and Input to Decision and Update ##########
High level loop
1. Observe the world event or input. This can be a player action, an NPC action, or a timed tick.
2. Build context. Retrieve the NPC state, the recent transcript, and a few long term summaries that are relevant.
3. Route into a LangGraph node. This node packages the context, applies rules, and can call tools before the LLM.
4. Call the LLM to evaluate options. The LLM scores or selects a choice from the allowed action set.
5. Execute the chosen action. This can be a dialogue line, a state change, or a tool call like give item.
6. Update memory and relationships with small bounded deltas. Summarize if needed.
7. Schedule follow ups such as cool downs or delayed reactions. Then wait for the next input.

ASCII flowchart with tools and feedback loop

    +------------------+           +----------------+
    |   NPC State      |           |  User or NPC   |
    | backstory stats  |           |   Input Event  |
    | goals memory     |           | action dialog  |
    +---------+--------+           +-------+--------+
              \                          /
               \                        /
                \                      /
                 v                    v
             +-------------------------------+
             |   LangGraph Context Builder   |
             | retrieve memories             |
             | select long term summaries    |
             | enforce world rules           |
             | attach current event          |
             +---------------+---------------+
                             |
                             v
             +-------------------------------+
             |   Policy and Tool Gate        |
             | allowed actions for this node |
             | pre tools such as             |
             |   fetch inventory             |
             |   find location               |
             |   look up schedule            |
             +---------------+---------------+
                             |
                             v
             +-------------------------------+
             |   LLM Decision Node           |
             | choose action from set        |
             | produce text if dialogue      |
             | score alternatives            |
             +---------------+---------------+
                             |
               +-------------+-----------+
               |                         |
               v                         v
    +----------------+          +------------------------+
    | Chosen Action  |          | State Update and Memory|
    | speak join etc |          | bounded stat deltas    |
    +--------+-------+          | relation adjust        |
             |                  | memory write and       |
             |                  | summarization          |
             |                  +-----------+------------+
             |                              ^
             |                              |
             +------------------------------+
                    feedback into next loop

Key tools
* Memory retrieve and write with salience ranking and TTL for short term
* Summarizer that compresses old dialogue into a first person fact such as Jerry is my friend and grew up fishing
* Relationship graph update that adjusts trust and affinity with clamps and cooldowns
* Inventory, location, and schedule helpers that feed the policy and LLM
* Safety and style filter for in world tone and lore

########## Data Structures ##########
NPC profile schema example JSON

{
  "id": "npc_alina",
  "name": "Alina Reed",
  "backstory": "Left the city to learn herbalism with her grandmother in the valley.",
  "traits": ["curious", "loyal", "soft_spoken"],
  "motivations": ["belonging", "mastery", "service"],
  "dreams": ["run a tea shop", "collect rare seeds"],
  "secrets": ["mother is from the rival guild"],
  "stats": {
    "loyalty": 35,
    "courage": 40,
    "mood": 55,
    "curiosity": 70
  },
  "relationships": {
    "npc_boris": {"trust": 30, "affinity": 45},
    "player": {"trust": 20, "affinity": 25}
  },
  "memories": {
    "short_term": [],
    "long_term": [
      "I enjoy quiet mornings at the well with Boris",
      "The player brought me lavender and listened to my plans"
    ]
  }
}

Relationship graph
* Directed or undirected edges between characters
* Edge weights include trust and affinity and respect (values from 0 to 100)
* Decay applies over time when there is no interaction
* Gossip can move small amounts of trust between non neighbors

Bounded updates
* Clamp every stat delta to a small range such as minus 5 to plus 5 per event
* Apply soft limits near extremes through a logistic curve so values slow near 0 and 100
* Use cooldown timers to prevent rapid oscillation
* Add light noise so equally repeated actions do not produce identical outcomes

########## Prompt Patterns ##########
Action selection prompt skeleton

You are an NPC named {name}. You are in a cozy town role playing game.
Stay in character. Keep responses short and grounded in the scene.
Facts about you
{backstory}
Traits and motivations
{traits}  {motivations}
Recent context
{recent_events}
Long term summaries
{long_term}
Allowed actions for this node
{allowed_actions}
Choose exactly one action and give a very short reason. If the action is speak include a single sentence of dialogue in quotes.

Example LLM output
action: SPEAK
reason: I want to thank the player for listening yesterday
line: "Thank you for the lavender. I have been trying new blends."

########## Minimal Code Sketch ##########
Python style with your header format and casual comments

########## npc.py ##########
class NPC:
    def __init__(self, profile):
        self.id = profile["id"]                              # store id
        self.name = profile["name"]                          # store name
        self.state = profile                                 # full state ref

    def view(self):
        return {
            "id": self.id,
            "name": self.name,
            "stats": self.state["stats"],
            "relations": self.state.get("relationships", {}),
        }                                                    # safe public view

########## memory.py ##########
class MemoryStore:
    def __init__(self):
        self.short = {}                                      # short term map
        self.long = {}                                       # long term map

    def add_event(self, npc_id, event):
        self.short.setdefault(npc_id, []).append(event)      # push event

    def summarize(self, npc_id):
        items = self.short.get(npc_id, [])[-8:]              # last window
        if not items:
            return None                                      # nothing to do
        summary = f"I remember {len(items)} recent moments with warmth."      # placeholder
        self.long.setdefault(npc_id, []).append(summary)     # store summary
        self.short[npc_id] = []                              # clear short term

########## relationships.py ##########
def adjust_relation(edge, d_trust=0, d_affinity=0):
    def clamp(x):
        return max(0, min(100, x))                           # clamp helper
    edge["trust"] = clamp(edge.get("trust", 50) + d_trust)   # update trust
    edge["affinity"] = clamp(edge.get("affinity", 50) + d_affinity)   # update
    return edge                                              # return edge

########## graph_nodes.py ##########
# Node skeleton for LangGraph
def node_talk(context):
    allowed = ["SPEAK", "IGNORE"]                            # allowed set
    # pre tools could fetch inventory or location here         # pre tools
    llm_out = call_llm(context, allowed)                     # choose action
    if llm_out["action"] == "SPEAK":
        say(llm_out["line"])                                 # perform action
        context["effects"].append(("affinity", +2))          # small bump
    return llm_out                                           # pass along

########## update.py ##########
def apply_effects(npc, effects):
    for stat, delta in effects:
        val = npc.state["stats"].get(stat, 50)               # default mid
        val = max(0, min(100, val + max(-5, min(5, delta)))) # clamp delta
        npc.state["stats"][stat] = val                       # write back

########## UI and Demo ##########
We want to see something while the code runs. Two paths are suggested.

Path one Streamlit simple web app
* Panel with NPC cards and stats
* Live dialogue feed with filters by character
* Relationship graph using NetworkX with spring layout
* Controls to start stop speed up and reset the sim

Path two Textual or Rich console app
* Live layout with three columns
* Left is NPC list with stats
* Middle is dialogue log
* Right is relation matrix or tiny graph

########## Development Setup ##########
Local in VS Code
1. Create a Python 3.11 environment
   python -m venv .venv
   source .venv/bin/activate on mac or linux
   .venv\Scripts\activate on windows

2. Install core libraries
   pip install langgraph langchain pydantic networkx streamlit rich textual fastapi uvicorn

3. Run the demo
   streamlit run soulscript_demo.py
   or
   python -m textual soulscript_tui.py

4. Optional local LLM for cost control
   Add an OpenAI compatible server or run an open model locally and point the LangChain client to it

########## Recommended Tools and Libraries ##########
Core
* LangGraph orchestration backbone
* LangChain prompt and memory helpers
* Pydantic for data validation
* NetworkX for relationship graphs
* SQLite via built in sqlite3 for persistence
* Streamlit for a simple web demo
* Rich and Textual for a console interface

Integrations
* FastAPI for a small REST layer and websockets
* Unity C sharp plugin calling the REST API
* Unreal Python API and blueprints that call the REST layer

Data and eval
* Pandas for logs and metrics
* Matplotlib for simple charts inside the demo

########## Roadmap ##########
Phase 1 convincing demo
* Small town sandbox with five to ten NPCs
* NPC to NPC chats with evolving trust and affinity
* Memory logging and periodic summarization
* Simple Streamlit or Textual UI
* Export of a run log for later review

Phase 2 developer tools
* NPC profile builder GUI
* State inspector for stats memories and relations
* Visual LangGraph editor with drag and drop nodes
* Save and load that is stable across versions

Phase 3 integration hooks
* Unity and Unreal bindings
* Dialogue exporter to file for cutscenes
* Stable API between engine and SoulScript

Phase 4 advanced depth
* Dreams and secrets that steer behavior
* Gossip and rumor network
* Factions and group alignment
* Emotional memory decay and forgiveness
* Procedural quest suggestions from conflicts and needs

Long term potential
* Emergent storytelling across a whole town
* Party dynamics that feel personal
* Cross game continuity with import and export of souls
* Community marketplace for sharable NPC profiles

########## Metrics and Evaluation ##########
What to track
* Conversation count and talk ratio per pair
* Trust and affinity trajectories over time
* Choice distribution across nodes
* Repetition rate in dialogue lines
* Player perceived variety via simple surveys

Health checks
* Token use and latency per call
* Cache hit rate for retrieval
* Memory summarization frequency
* Error rate in tool calls

########## Cost and Performance ##########
Reduce tokens
* Retrieve fewer but more relevant memories with embeddings and salience tags
* Use system prompts that are short and specific
* Summarize early and often and prune raw logs

Improve speed
* Parallelize independent NPC loops with a small scheduler
* Cache tool results for the current tick
* Use low temperature for policy steps

########## Safety and Style ##########
Lore and tone filter
* Keep speech in world and aligned with setting
* Avoid modern slang unless the world expects it
* Keep dialogue short and legible

Content safety
* Provide a developer toggle for stricter output
* Block topics that do not fit the rating of the game

########## File Layout Suggestion ##########
soulscript/
  core/
    npc.py
    memory.py
    relationships.py
    policy.py
    graph_nodes.py
    tools.py
  demo/
    soulscript_demo.py
    ui_streamlit.py
    ui_textual.py
  engine_api/
    server.py  FastAPI service
  data/
    profiles/
    saves/
  tests/
    test_memory.py
    test_policy.py
    test_updates.py
  README.txt  this file

########## Minimal API for Engines ##########
POST /tick
body includes events and list of active NPC ids
returns chosen actions for this tick

POST /npc_state
body includes npc id
returns public view of state and current goals

POST /dialogue
body includes speaker text and listener ids
returns rendered lines and any state changes

########## Closing ##########
Most NPC systems are static and predictable. SoulScript makes them grow. The plan is a convincing demo you can watch, then real developer tools, then engine bindings so you can drop these agents into your worlds. If you want to build something that feels alive, start here.

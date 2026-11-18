# robot-fleet-inference

A small toolkit for selling remote policy inference to robot fleets, built from public information about [Reactor](https://www.reactor.inc/robotics)'s Remote Policy API.

**Live calculator:** https://abdullah-coder25.github.io/Robot-Fleet-Inference/calculator/

It has three parts:

| Part | What it does |
|---|---|
| [`calculator/`](calculator/index.html) | "Can this robot run from the cloud?" Pick a hosted policy, a task and a fleet. It checks the task's latency budget against the network round trip plus the model's p99, prices the fleet per robot-hour, and shows how full a GPU server has to be before the price covers the GPUs. |
| [`data/accounts.csv`](data/accounts.csv) | Forty-seven net-new robotics accounts, none already partnered with Reactor (operators, model makers, channels, data fleets), each with the signals used for scoring, a suggested first call, the first call's work email and two public sources. |
| [`score.py`](score.py) | Scores and ranks the accounts, then writes `out/ranked.csv` and `out/pipeline.md`. Change the weights from the command line to disagree with them. |

## The idea behind it

Reactor publishes enough to do the unit economics:

- **Price:** Praxis-1 costs $0.030 per session-minute, or $1.80 per robot-hour ([docs](https://docs.reactor.inc/robotics/overview)).
- **Capacity:** on two B200s, Reactor serves 16 robots under a 250 ms p99, or one robot at 62 ms ([Praxis-1 post](https://www.reactor.inc/blog/praxis-1)).
- **GPU rent:** B200s rent for a median $8.19 per GPU-hour as of Nov 9, 2026 ([getdeploying](https://getdeploying.com/reference/cloud-gpu/nvidia-b200)).

Put together, a shared server covers its GPU rent only when about 9 of its 16 slots are in use at once. Because Reactor places capacity near each deployment, robots in the same metro can share a server and robots in a new metro cannot. So the pipeline is scored partly on metro density, not only on account size.

Latency decides which deals are winnable now. Reactor's own [DreamZero sweep](https://www.reactor.inc/blog/dreamzero) found pick-and-place tolerates ~924 ms of delay and insertion ~479 ms, while stacking and sorting tolerate ~150 ms. The shared tier (270 ms p99 plus a ~30 ms round trip) covers the first group. The second needs the dedicated tier.

## Run it

```bash
python score.py                                   # default weights
python score.py --weights offboard=3 lockin=-2    # your own weights
open calculator/index.html                        # or serve the repo with GitHub Pages
```

No dependencies beyond the Python standard library. The calculator is one static HTML file.

## Scoring

Each signal is a 0 to 3 judgment recorded in the CSV next to its sources:

| Signal | Weight | Meaning |
|---|---|---|
| `offboard` | 2.0 | Already runs inference off the robot, or every robot is cloud-connected |
| `model_match` | 2.0 | Runs a model Reactor hosts, or an open or in-house model it could bring to Serve |
| `latency_fit` | 1.5 | Its core task fits the shared tier's latency |
| `robot_hours` | 1.5 | Fleet size times hours per day, which is what fills servers |
| `density` | 1.0 | Computed: how many other targets share its metro |
| `warmth` | 1.0 | Relationship strength: 3 = I know the first call personally; 2 = met someone there in person, or a direct VC intro to a founder; 1 = indirect path (mentor's podcast guest, event overlap, reshared by Everett); 0 = cold |
| `lockin` | -1.5 | Its policy is served through another provider's API today |

## Honesty notes

- Everything comes from public pages linked in the CSV and the calculator. Figures I computed are labeled as estimates.
- Reactor's real GPU costs, regions and margins are private. The calculator uses market rents, so treat its margins as directional.
- Robots-per-server is published only for Praxis-1. Other policies default to a placeholder of 1; edit it.
- The onboard comparison is a memory-bandwidth floor (weights read per call divided by bandwidth), not a benchmark.
- Contacts come from LinkedIn and Apollo search results. Work emails come from Apollo enrichment (work emails only, no personal addresses), with Apollo's `email_status` recorded next to each; all 40 were marked verified by Apollo on Nov 18, 2026, and the Ai2 contact's address is his University of Washington one. No profiles were scraped and no one has been contacted.

Built by Abdullah Usman, November 2026. Not affiliated with Reactor.

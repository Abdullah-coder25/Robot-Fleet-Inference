# robot-fleet-inference

Two calculators plus 170 scored leads for Reactor: 48 robotics fleets for remote policy inference, and 122 more opened up by NVIDIA Cosmos on Reactor and by real-time generative media and gaming, built from public information about [Reactor](https://www.reactor.inc/robotics)'s Remote Policy API. Each lead has a first-call contact, work email, why-now reason and two sources; the ranked list is in [`out/ranked.csv`](out/ranked.csv).

**Why move now:** Fleets rarely rewire a working policy endpoint, so the next 12 months are a land grab. The calculator answers a robotics team's first question and Reactor's: does this deal fill a server? From day one, I would run GTM on these 48 leads, warm intros first, to land Reactor many robotics customers.

**Live calculator:** https://abdullah-coder25.github.io/Robot-Fleet-Inference/calculator/

**Evaluation calculator (Cosmos on Reactor vs real robots):** https://abdullah-coder25.github.io/Robot-Fleet-Inference/calculator/eval.html

**Integration example:** [MuJoCo + Cosmos Transfer on Reactor](cosmos-transfer-mujoco/), a closed-loop simulator integration with videos, measured latency and cost, and a partner quickstart.

It has seven parts:

| Part | What it does |
|---|---|
| [`calculator/`](calculator/index.html) | "Can this robot run from the cloud?" Pick a hosted policy, a task and a fleet. It checks the task's latency budget against the network round trip plus the model's p99, prices the fleet per robot-hour, and shows how full a GPU server has to be before the price covers the GPUs. |
| [`calculator/eval.html`](calculator/eval.html) | "Test in Cosmos first, or on real robots?" Enter an evaluation plan and compare the weekly cost, time and human hours of running every trial on real robots against screening in NVIDIA Cosmos on Reactor and confirming the best versions on hardware. |
| [`cosmos-transfer-mujoco/`](cosmos-transfer-mujoco/) | A working integration: MuJoCo running closed loop through NVIDIA Cosmos3 Nano Sim Transfer on Reactor, with side-by-side video, appearance variants, measured step time and cost, and a first-working-build checklist for partners. |
| [`data/accounts.csv`](data/accounts.csv) | The input: 48 net-new robotics accounts, none already partnered with Reactor (operators, model makers, channels, data fleets). Columns: Account, First call, Title, Connection Notes, Email, Segment, Metro, Why now, the six 0 to 3 signals (Offboard, Model match, Latency fit, Robot-hours, Warmth, Lock-in), Source 1 and Source 2. |
| [`data/robotics_cosmos.csv`](data/robotics_cosmos.csv), [`data/gen_media.csv`](data/gen_media.csv), [`data/gaming.csv`](data/gaming.csv) | The expansion: 63 robotics accounts that can now use Cosmos on Reactor (foundation model teams, eval and simulation companies, data collection, labs, hardware makers, AV and surgical), 39 generative media accounts (video model labs, real-time avatars, creative apps, world-model labs) and 20 gaming accounts. Each row has a first-call contact, work email, why-now reason, 0 to 3 signals and two sources; `score.py` ranks them into `out/ranked_<segment>.csv`. Companies Reactor already lists as customers, partners or hosted models are excluded. |
| [`score.py`](score.py) | Works out Metro density, Score and Rank, then writes the two files in `out/`. Change the weights from the command line to disagree with them. |
| [`out/`](out/) | The output. [`ranked.csv`](out/ranked.csv) is every account, ranked, with Rank, Score and Metro density added to the input columns. [`pipeline.md`](out/pipeline.md) is a shorter, readable version with the metro counts on top. |

## The idea behind it

Reactor publishes enough to do the unit economics:

- **Price:** Praxis-1 costs $0.030 per session-minute, or $1.80 per robot-hour ([docs](https://docs.reactor.inc/robotics/overview)).
- **Capacity:** on two B200s, Reactor serves 16 robots under a 250 ms p99, or one robot at 62 ms per call (p99 64 ms) ([Praxis-1 post](https://www.reactor.inc/blog/praxis-1)).
- **GPU rent:** B200s rent for a median $7.99 per GPU-hour across 27 providers as of Nov 17, 2026 ([getdeploying](https://getdeploying.com/reference/cloud-gpu/nvidia-b200)).

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

Each signal is a 0 to 3 judgment, backed by the two sources in its row:

| Signal | Weight | What 0 to 3 means |
|---|---|---|
| Offboard | 2.0 | Does it already run policies off the robot? 3 = cloud, teleop or remote humans in the loop today; 0 = everything runs onboard. |
| Model match | 2.0 | Does it run a model Reactor can host or serve? 3 = an open model or its own large model that needs serving; 0 = locked to on-device. |
| Latency fit | 1.5 | Does its work fit the shared tier's ~300 ms round trip? 3 = pick-and-place, folding, unloading; 1 = precision work like assembly. |
| Robot-hours | 1.5 | How much server time it could fill. 3 = a large fleet or long shifts today; 0 = no robots shipping yet. |
| Metro density | 1.0 | Computed, not entered: 3 = 5 or more targets in the metro; 2 = 2 to 4; 1 = alone; 0 = metro unknown ("n/a"). |
| Warmth | 1.0 | 3 = I know the first call personally; 2 = met someone there in person, or a direct VC intro to a founder; 1 = indirect path (mentor's podcast guest, event overlap); 0 = cold. |
| Lock-in | -1.5 | Tied to another lab's model or stack, such as Physical Intelligence or Gemini Robotics. Subtracts. |

Score = 100 x (weighted sum) / 27, the highest possible sum, rounded half up. Ties share a rank (1, 1, 3), and tied accounts keep their order from the input file.

The expansion files use signals that fit each market, with the same 0 to 3 scale and the same formula (score = 100 x weighted sum / highest possible sum):

| Segment | Signals and weights |
|---|---|
| Robotics + Cosmos | World-model fit 2, Real-time need 2, Volume 1.5, Warmth 1, Lock-in -1.5 |
| Gen media and gaming | Real-time need 2, Own models 2, Scale 1.5, Infra pain 1.5, Warmth 1, Lock-in -1.5 |

## Honesty notes

- Facts about each account come from the public pages linked in its row; Connection Notes and Warmth are my own relationships. Figures I computed are labeled as estimates.
- Reactor's real GPU costs, regions and margins are private. The calculator uses market rents, so treat its margins as directional.
- Robots-per-server is published only for Praxis-1. For the other policies, the calculator starts from the fewest robots per server that would cover the GPU rent at Reactor's price, and marks that as an assumption you can change.
- The onboard comparison is a memory-bandwidth floor (weights read per call divided by bandwidth), not a benchmark.
- Contacts come from LinkedIn and Apollo search results. Work emails come from Apollo enrichment (work emails only, no personal addresses), all marked verified by Apollo on Nov 18, 2026, and the Ai2 contact's address is his University of Washington one. No profiles were scraped and no one has been contacted.

Built by Abdullah Usman, November 2026. Not affiliated with Reactor.

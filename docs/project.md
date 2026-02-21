# Automated Reflectivity Analysis

We are expanding the capabilities of the reflectivity analysis agent in https://github.com/neutrons-ai/aure.git

Think of it as AuRE v2. We will not work in the AuRE repo, but start a new project in this repo.

To do this, we will use OpenHands for coding! So our work here is just to set up the
project using OpenHands, and then launch the coding project.

I need 4 worker agents interacting together: a project manager, two coders, and a tester.

Our LLM service will be local, using the OpenAI protocol.

The following section explains the reference data for test cases, and the rest of this
document is an implementation plan.


## Reference data
For test cases using real data, we have a curated set in $USER/git/experiments-2024/val-sep24/models/corefined. Each data set is in a separate folder. In each folder, the files of interest are:
    - <model-name>-refl.dat: this is the data, with the 'theory' column being the ground truth model.
    - <model-name>.err: this lists the ground truth model parameters, with uncertainties (since this is real data).

The input data for these fits are in $USER/git/experiments-2024/val-sep24/data. 
The mapping of input data to fit models is the following:

| Comparison | Cu Substrate | Condition| Runs |
| :-----------|--------------|----------| --- |
|Cycling vs. sustained current | D | Cycling | 213032 & 213036 |
|  -  | I | Sustained| 213082 & 213086 |
|Varying ethanol concentration | F | 0% ethanol | 213046 & 213050 |
|  -  | D | 1% ethanol | |
|  -  | E | 2% ethanol | 213039 & 213043 |
|Deuteration Contrast | D | d8-THF + EtOH |
|  -  | G | d8-THF + d6-EtOH | 213056 & 213060 |
|  -  | M | THF + d6-EtOH | 213136 & 213140 |
|  -  | K | THF + EtOH | 213110 & 213114 |
|Concentration dependence | D | 0.2 M Cabhfip | |
|  -  | L | 0.1 M Cabhfip | 213126 & 213130 |


The input of a test case should only be an input file in $USER/git/experiments-2024/val-sep24/data and the following description:

```
Copper main layer (50 nm) on a titanium sticking layer (5 nm) on a silicon substrate.
The ambiant medium is most likely dTHF electrolyte, but may be THF.
The reflectivity was measured from the back of the film, with the incoming beam coming from the silicon side.
```



## Phase 1: Local Infrastructure Setup

Since you are using a local OpenAI-compatible protocol, OpenHands uses **LiteLLM** to bridge the connection.

1. **Start your Local LLM Server:** Our LLM model server is LLM_BASE_URL=http://localhost:8555/v1

2. **Launch OpenHands via Docker:**
```bash
docker run -it --rm \
    --pull=always \
    -e SANDBOX_USER_ID=$(id -u) \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands-state:/.openhands-state \
    -p 3000:3000 \
    ghcr.io/all-hands-ai/openhands:0.20

```

3. **Connect the Model:**
* In the OpenHands UI (Settings), set **LLM Provider** to `OpenAI`.
* Set **Base URL** to your local endpoint (e.g., `http://host.docker.internal:11434/v1`).
* Set **Model** to `openai/gpt-oss-120b`.



---

## Phase 2: Defining Your Personas (Microagents)

OpenHands uses **Microagents** to define specialized behaviors without writing complex Python logic. Create a folder named `.openhands/microagents/` in your project root and add these four `.md` files:

### 1. `pm.md` (The Orchestrator)

> **Role:** Project Manager
> **Responsibilities:** Break down the user prompt into sub-tasks. Use the `spawn` and `delegate` tools to assign work to `dev-alpha`, `dev-beta`, and `tester`. You are responsible for the final merge.
> **Constraint:** Never write code yourself; always delegate.

### 2. `dev-alpha.md` & `dev-beta.md` (The Workers)

> **Role:** Senior Developer
> **Responsibilities:** Implement features assigned by the PM. Write clean, modular code. You have access to the terminal and filesystem.
> **Workflow:** Always create a new branch for your task. Notify the PM when your code is ready for testing.

### 3. `tester.md` (The Quality Gate)

> **Role:** QA Engineer
> **Responsibilities:** Verify all code changes. You must run the local test suite (e.g., `pytest` or `npm test`).
> **Goal:** Do not approve a task unless 100% of tests pass. Report specific error logs back to the Developers if they fail.

---

## Phase 3: Executing the Multi-Agent Workflow

OpenHands uses **Event Streaming** for delegation. To start your team, you only need to talk to the **PM**.

1. **Initialize the Session:** In the OpenHands chat, select the **PM** persona.
2. **The Prompt:**
> "PM, we are building a [Project Description]. Please spawn two developers and one tester. I need you to coordinate the development of the auth module and the database schema. Ensure the tester verifies everything before you report back to me."



### What happens behind the scenes:

* **Spawning:** The PM agent calls `spawn(["dev-alpha", "dev-beta", "tester"])`.
* **Delegation:** The PM sends specific instructions: `delegate({"dev-alpha": "Create SQLAlchemy models", "dev-beta": "Implement JWT logic"})`.
* **Parallelism:** Both developers work in their own isolated Docker sessions.
* **Handoff:** When Dev-Alpha finishes, the PM sends that specific file path to the **Tester** to run the pre-defined test suite.

---

## Phase 4: Monitoring and Safety

* **Workspace Isolation:** Each agent operates in the same project directory but their "thought process" and command history are isolated.
* **Human-in-the-Loop:** You can watch the "Event Stream" in the OpenHands UI. If the PM gets stuck or the local LLM starts hallucinating a bash command, you can intervene in the terminal directly.


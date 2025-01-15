# MARS - Predicting Correctness from Model Internals

## Taskfile

The `Taskfile.yml` contains a list of tasks that can be run using the `task` command.

To see a list of available tasks, run the following command:

```bash
task --list
task: Available tasks for this project:
* dep-lock:              Update the project's lockfiles from the requirements-<env>.in files
* format:                Format the code using black and isort
* format-datasets:       Prepare benchmark datasets for question-answering
```

The `task` binary is already available in the DevContainer. To install it on your local machine, follow the instructions [here](https://taskfile.dev/installation).

## Downloading HF resources

> Remember to store on pod's persistent volume mounted at `/workspace`.

- Download `huggingface-cli`: `pip install -U "huggingface_hub[cli]"`.
- Login with HF token before downloading for gated resources (`huggingface-cli login`).
- Download model: `huggingface-cli download --local-dir /workspace/models/<local_model_id> <model_id_hf>`.

Resources directory structure:

- Models: `/workspace/models`
- Datasets: `/workspace/datasets`

## Experiments

Sample number: 1000 samples?

1. PCA / tSNE on activations at ${l_i, ..., l_n}$, where $l_i \in$ residual stream transformer layers, on settings:
    - Prompt only
    - Full generation (prompt + generated answer)
    - (opt) CoT setting: (prompt + CoT, but no answer)
        - Potential limitation: how to detect end of CoT / start of answer in model generations
2. Visualize plots in search of clusters across $n$ layers with clear separation among correct / incorrect generations. With the identified layer, train a linear classifier for the previously defined settings.
    - Single parameter: answer correctness

Evaluation: compare model generations against ground truth to obtain labels. 

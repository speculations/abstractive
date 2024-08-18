<br>

## Environments

### Remote Development

For this Python project/template, the remote development environment requires

* [Dockerfile](../.devcontainer/Dockerfile)
* [requirements.txt](../.devcontainer/requirements.txt)

An image is built via the command

```shell
docker build . --file .devcontainer/Dockerfile -t text
```

On success, the output of

```shell
docker images
```

should include

<br>

| repository | tag    | image id | created  | size     |
|:-----------|:-------|:---------|:---------|:---------|
| text       | latest | $\ldots$ | $\ldots$ | $\ldots$ |


<br>

Subsequently, run a container, i.e., an instance, of the image `text` via:

<br>

```shell
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 -w /app 
	--mount type=bind,src="$(pwd)",target=/app text
```

<br>

Herein, `-p 10000:8888` maps the host port `10000` to container port `8888`.  Note, the container's working environment, i.e., -w, must be inline with this project's top directory.  Additionally

* --rm: [automatically remove container](https://docs.docker.com/engine/reference/commandline/run/#:~:text=a%20container%20exits-,%2D%2Drm,-Automatically%20remove%20the)
* -i: [interact](https://docs.docker.com/engine/reference/commandline/run/#:~:text=and%20reaps%20processes-,%2D%2Dinteractive,-%2C%20%2Di)
* -t: [tag](https://docs.docker.com/get-started/02_our_app/#:~:text=Finally%2C%20the-,%2Dt,-flag%20tags%20your)
* -p: [publish](https://docs.docker.com/engine/reference/commandline/run/#:~:text=%2D%2Dpublish%20%2C-,%2Dp,-Publish%20a%20container%E2%80%99s)

<br>

Get the name of the running instance of ``text`` via:

```shell
docker ps --all
```

Never deploy a root container, study the production [Dockerfile](../Dockerfile); cf. [/.devcontainer/Dockerfile](../.devcontainer/Dockerfile)

<br>

### Remote Development & Integrated Development Environments

An IDE (integrated development environment) is a helpful remote development tool.  The **IntelliJ
IDEA** set up involves connecting to a machine's Docker [daemon](https://www.jetbrains.com/help/idea/docker.html#connect_to_docker), the steps are

<br>

> * **Settings** $\rightarrow$ **Build, Execution, Deployment** $\rightarrow$ **Docker** $\rightarrow$ **WSL:** {select the linux operating system}
> * **View** $\rightarrow$ **Tool Window** $\rightarrow$ **Services** <br>Within the **Containers** section connect to the running instance of interest, or ascertain connection to the running instance of interest.

<br>

**Visual Studio Code** has its container attachment instructions; study [Attach Container](https://code.visualstudio.com/docs/devcontainers/attach-container).


<br>
<br>

## Code Analysis

The GitHub Actions script [main.yml](../.github/workflows/main.yml) conducts code analysis within a Cloud GitHub Workspace.  Depending on the script, code analysis may occur `on push` to any repository branch, or `on push` to a specific branch.

The sections herein outline remote code analysis.

### pylint

The directive

```shell
pylint --generate-rcfile > .pylintrc
```

generates the dotfile `.pylintrc` of the static code analyser [pylint](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html).  Analyse a directory via the command

```shell
python -m pylint --rcfile .pylintrc {directory}
```

The `.pylintrc` file of this template project has been **amended to adhere to team norms**, including

* Maximum number of characters on a single line.
  > max-line-length=127

* Maximum number of lines in a module.
  > max-module-lines=135


<br>


### pytest & pytest coverage

The directive patterns

```shell
python -m pytest tests/{directory.name}/...py
pytest --cov-report term-missing  --cov src/{directory.name}/...py tests/{directory.name}/...py
```

for test and test coverage, respectively.


<br>


### flake8

For code & complexity analysis.  A directive of the form

```bash
python -m flake8 --count --select=E9,F63,F7,F82 --show-source --statistics src/...
```

inspects issues in relation to logic (F7), syntax (Python E9, Flake F7), mathematical formulae symbols (F63), undefined variable names (F82).  Additionally

```shell
python -m flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics src/...
```

inspects complexity.

<br>
<br>

## Notes

### Tune

> [!NOTE]
> [Tuners](https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html) can also be used to launch hyperparameter tuning without using Ray Train, e.g., [ray.train.torch.TorchTrainer](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html)

For the ray board

```bash
127.0.0.1:8265
```

If the ...

```shell
tensorboard --logdir /tmp/ray/session_2024-08-18_12-20-14_323079_69424/artifacts/2024-08-18_12-20-26/tuning/driver_artifacts
```

Note
* RunConfig(storage_path='', ...) for [specifying the parent directory](https://docs.ray.io/en/latest/tune/tutorials/tune-output.html) of the trials data

<br>

### Address



# ... steps & epochs
max_steps_per_epoch = self.__source['train'].shape[0] // (variable.TRAIN_BATCH_SIZE * variable.N_GPU)
max_steps = max_steps_per_epoch * self.__n_epochs


<br>

### Warnings

Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)


Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())




<br>
<br>

## References

* Ray Board
  * [ray.init()](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html)
  * [Getting Started](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
* Distributed Training
  * [Distributed Communication](https://docs.w3cub.com/pytorch/distributed.html)
  * [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
* Modelling
  * [Hyperparameter Tuning with Ray Tune](https://docs.ray.io/en/latest/train/user-guides/hyperparameter-optimization.html)
  * [Getting Started with Ray Tune](https://docs.ray.io/en/latest/tune/getting-started.html)
  * [train hyperparameter search](https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html)
  * [Logging and Outputs in Tune](https://docs.ray.io/en/latest/tune/tutorials/tune-output.html)
  * [Tune Experiments](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html)

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>

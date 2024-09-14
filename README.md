<br>

In Progress: Abstractive Text Summarisation

<br>

### Designs

#### Design I

A container/instance of a repository image will expect a string argument.  The
argument will determine the model development activity that the instance will focus on.

```bash
match architecture:
  case 't5':
    src.modelling.t5.steps.Steps(...)
  case 'pegasus':
    src.modelling.pegasus.steps.Steps(...)
  case 'bart':
    src.modelling.bart.steps.Steps(...)
  case _:
    return 'Unknown architecture'
```

<br>

#### Design II

A repository focusing on a single model architecture only.  **This is a Design II repository.**

Approach: huggingface.co transformers $+$ ray.io trainer & tuner $\rightarrow$

* TorchTrainer: receives
  * transformers.Seq2SeqTrainer
  * data
* Tuner: for setting
  * parameter space: training loop configuration, scaling/resources configuration
  * TuneConfig: objective function, optimisation scheduler, etc.
  * RunConfig: storage, checkpoints
* **Elsewhere**
  * A set-up for hyperparameter optimisation via [Population Based Training](https://docs.ray.io/en/latest/tune/api/schedulers.html#population-based-training-tune-schedulers-populationbasedtraining) 


<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>

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

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>

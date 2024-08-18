# PyTorch
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Temporary
ARG GID=3333
ARG UID=$GID

# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
COPY .devcontainer/requirements.txt /app
RUN groupadd --system automata --gid $GID && \
    useradd --system automaton --uid $UID --gid $GID && \
    apt update && apt -q -y upgrade && apt -y install sudo && sudo apt -y install graphviz && \
    pip install --upgrade pip && \
    pip install --requirement /app/requirements.txt --no-cache-dir

# Specific COPY
COPY src /app/src
COPY config.py /app/config.py

# Port
EXPOSE 8050

# Reader
USER automaton

# ENTRYPOINT
ENTRYPOINT ["python"]

# CMD
CMD ["src/main.py"]
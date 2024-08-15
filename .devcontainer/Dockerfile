# Base Image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime


# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app

COPY /.devcontainer/requirements.txt /app

RUN pip install --upgrade pip && \
	pip install --requirement /app/requirements.txt --no-cache


# Port
EXPOSE 8050


# ENTRYPOINT
ENTRYPOINT ["bash"]
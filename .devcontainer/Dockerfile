# VSCode Base Image
ARG PYTHON_VERSION="3.8"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${PYTHON_VERSION}

# install opencv
RUN apt-get update && \
    apt-get -y install libopencv-dev

# Install Poetry
ARG POETRY="1.2.2"
RUN pip install poetry==${POETRY}
RUN pip install keyrings.google-artifactregistry-auth==1.0.0

# add credentials to install private repos
ARG GOOGLE_APPLICATION_CREDENTIALS
COPY ${GOOGLE_APPLICATION_CREDENTIALS} /${GOOGLE_APPLICATION_CREDENTIALS}

# Install python dependencies using poetry
COPY pyproject.toml ./
RUN export GOOGLE_APPLICATION_CREDENTIALS=/${GOOGLE_APPLICATION_CREDENTIALS} \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

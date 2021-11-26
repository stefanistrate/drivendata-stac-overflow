FROM stefanistrate/deeplearning:py3.9-tensorflow2.7
ARG PIN_DEPENDENCIES=false

# Set up the current project.
WORKDIR /gridai/project
COPY . .
RUN if $PIN_DEPENDENCIES ; \
    then \
    python -m pip install --no-cache-dir --upgrade -r requirements.txt \
    && python -m pip install --no-cache-dir --upgrade -e . --no-deps \
    ; \
    else \
    python -m pip install --no-cache-dir --upgrade -e . \
    ; \
    fi

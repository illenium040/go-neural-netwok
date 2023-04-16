#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PROJECT_DIR=$(cd "$SCRIPT_DIR"/../ &> /dev/null && pwd )

go build -o ./bin/neural-network "$PROJECT_DIR"/src

"$PROJECT_DIR"/bin/neural-network
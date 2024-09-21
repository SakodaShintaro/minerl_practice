#!/bin/bash
set -eux

cd $(dirname $0)

docker build -t ${1} .

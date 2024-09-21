#!/bin/bash
set -eux

sudo apt-get update
sudo apt-get install -y software-properties-common

sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk

pip3 install git+https://github.com/minerllabs/minerl

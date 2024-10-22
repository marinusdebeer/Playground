#!/bin/bash
source ~/.bashrc
nvm exec 20 nohup node index.js > server.log 2>&1 &

#!/bin/bash
source ~/.bashrc

# Use the exact version explicitly
nvm exec 20.18.0 nohup node index.js > server.log 2>&1 &

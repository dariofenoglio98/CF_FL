#!/bin/bash

echo "Starting server"
python server.py --rounds 50 &  # set number of rounds HERE
sleep 3 # Sleep for 3s to give the server enough time to start

for i in $(seq 1 2); do   # set number of clients HERE
    echo "Starting client $i"
    python client.py --id "${i}" &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
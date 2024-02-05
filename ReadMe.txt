Server.py creates a server to which clients can connect 
       --rounds: number of training rounds, default 20 

Client.py creates a generic client with id --id (link with a respective dataset)
        and it automatically connects to the server
        --id: specifies the artificial data partition
        --data_type: random or cluster partitioin of the data

RUN:
1. run.sh: automatically runs all codes (server and clients). Set parameters training inside
2. python server.py --rounds 200
   python client.py --id 1 --data_type "random"
   python client.py --id 2 --data_type "random"
   python client.py --id 3 --data_type "random"
   ...

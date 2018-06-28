#!/bin/bash

event_name="aoc-status"
api_key="cdgfxgGqbIS7A9mdYAouB2"

if [ $# -eq 3 ]
  then
    project=$1
    status=$2
    info=$3   
else
    echo Usage: PROJECT STATUS INFO
    exit 1
fi

reminder=$1
echo "Sending notification for $project with status $status and info $info"

curl --silent --show-error -X POST -H "Content-Type: application/json" --data '{"value1": "'"$project"'","value2": "'"$status"'","value3": "'"$info"'"}' \
https://maker.ifttt.com/trigger/$event_name/with/key/$api_key

echo .
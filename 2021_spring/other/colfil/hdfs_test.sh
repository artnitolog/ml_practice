#!/bin/bash

fallocate -l 100M /test.txt
hdfs dfs -mkdir /temp /logs
hdfs dfs -put /test.txt /temp
hdfs dfs -ls /temp/test.txt
hdfs dfs -mv /temp/test.txt /logs

hdfs dfs -setrep -w 1 /logs/test.txt
hdfs dfs -cp /logs/test.txt /logs/test2.txt
hadoop distcp /logs /logs2
hdfs dfs -chmod 0600 /logs2/test2.txt
hdfs dfs -ls /logs2

hdfs dfs -du -h /
hdfs dfs -rm -r /logs
hdfs fsck /logs2
hdfs dfsadmin -report
hdfs dfs -get /logs2/test2.txt /

hdfs dfs -appendToFile test2.txt /logs2/test.txt
hdfs dfs -du -h /logs2

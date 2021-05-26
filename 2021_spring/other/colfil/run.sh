# Declare paths
LOCAL_PATH=`dirname "$0"`
HPATH="/colfil"
INPUT_HADOOP_DIR="${HPATH}/input"
OUTPUT_HADOOP_DIR="${HPATH}/output"
HADOOP_STREAMING_PATH="${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar"

# Remove existing I/O directories
hdfs dfs -test -d ${HPATH}
if [ $? -eq 0 ];
  then
    echo "Remove (hdfs) ${HPATH}"
    hdfs dfs -rm -r ${HPATH}
fi
test -d ${LOCAL_PATH}/data/output
if [ $? -eq 0 ];
  then
    echo "Remove ${LOCAL_PATH}/data/output"
    rm -rf ${LOCAL_PATH}/data/output
fi

# Copy local input to HDFS
hdfs dfs -mkdir -p ${INPUT_HADOOP_DIR}
hdfs dfs -copyFromLocal ${LOCAL_PATH}/data/input/* ${INPUT_HADOOP_DIR}

# Change permissions
chmod -R 0777 ${LOCAL_PATH}/src

# Stage 1: Group by user
# Output format: r@items#ratings
echo "Running stage 1..."
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.reduces=4 \
  -D stream.num.map.output.key.fields=1 \
  -D stream.map.output.field.separator=@ \
  -D stream.reduce.input.field.separator=@ \
  -files ${LOCAL_PATH}/src \
  -mapper src/mapper_stage_1.py \
  -reducer src/reducer_stage_1.py \
  -input ${INPUT_HADOOP_DIR}/ratings.csv \
  -output ${OUTPUT_HADOOP_DIR}/stage_1 \

# Stage 2: Find sim(i,j)
# Output format: i@items#sims
echo "Running stage 2..."
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.reduces=4 \
  -D stream.num.map.output.key.fields=1 \
  -D stream.map.output.field.separator=@ \
  -D stream.reduce.input.field.separator=@ \
  -files ${LOCAL_PATH}/src \
  -mapper src/mapper_stage_2.py \
  -reducer src/reducer_stage_2.py \
  -input ${OUTPUT_HADOOP_DIR}/stage_1 \
  -output ${OUTPUT_HADOOP_DIR}/stage_2 \

# Stage 3: Aggregate r_ik, sim_ki
# Output format: i@j@r_ik,sim_ki
echo "Running stage 3..."
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.reduces=4 \
  -D stream.num.map.output.key.fields=1 \
  -D stream.map.output.field.separator=@ \
  -D stream.reduce.input.field.separator=@ \
  -files ${LOCAL_PATH}/src \
  -mapper src/mapper_stage_3.py \
  -reducer src/reducer_stage_3.py \
  -input ${OUTPUT_HADOOP_DIR}/stage_2,${INPUT_HADOOP_DIR}/ratings.csv \
  -output ${OUTPUT_HADOOP_DIR}/stage_3 \

# Stage 4: Estimate r_ui
# Output format: u@i@r_ui
echo "Running stage 4..."
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.reduces=9 \
  -D mapreduce.job.reduce.slowstart.completedmaps=1 \
  -D stream.num.map.output.key.fields=2 \
  -D stream.map.output.field.separator=@ \
  -D stream.reduce.input.field.separator=@ \
  -files ${LOCAL_PATH}/src \
  -mapper src/mapper_stage_4.py \
  -reducer src/reducer_stage_4.py \
  -input ${OUTPUT_HADOOP_DIR}/stage_3 \
  -output ${OUTPUT_HADOOP_DIR}/stage_4 \

# Stage 5 (final): make predictions
# Output format: u@rating1#title1@...@rating100#title100
echo "Running stage 5 (final)..."
hadoop jar ${HADOOP_STREAMING_PATH} \
  -D mapreduce.job.output.key.comparator.class=org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator \
  -D stream.num.map.output.key.fields=3 \
  -D stream.map.output.field.separator=@ \
  -D stream.reduce.input.field.separator=@ \
  -D mapreduce.map.output.key.field.separator=@ \
  -D mapreduce.partition.keypartitioner.options=-k1,1n \
  -D mapreduce.partition.keycomparator.options="-k1,1n -k3,3nr -k2,2" \
  -files ${LOCAL_PATH}/src,${LOCAL_PATH}/data/input/movies.csv \
  -mapper src/mapper_stage_5.py \
  -reducer src/reducer_stage_5.py \
  -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
  -input ${OUTPUT_HADOOP_DIR}/stage_4 \
  -output ${OUTPUT_HADOOP_DIR}/final \

hdfs dfs -copyToLocal ${OUTPUT_HADOOP_DIR} ${LOCAL_PATH}/data

hdfs dfs -rm -r ${HPATH}

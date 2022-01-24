# Undergraduate research project : ML4DB

Current assumption: there is no ad-hoc query.

## Instructions

### To simulate the workload : 

1. docker-compose up
2. update the pgSQL configaration file and restart the server
3. pgbench -U root -d postgres -i -s 5
4. pgbench -U root -d postgres -v -c 5 -T 1200 # 1200 implies 20 mins

### To run the optimizor

../src ./tester.py
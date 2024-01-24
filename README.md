# DTA-Model-Pipeline
Pipeline to go from OSM + flow data to an operational DTA model using e.g., DTALite or SUMO


## Workflow

1. Select sensors / associated road to include
2. Get road network in correct 
3. Get representative flow from road over defined time period (tbc, e.g., one day to begin)
4. Use SUMO tool to generate OD network and review
5. Convert OD matrix into correct format for DTALite
6. Simulate traffic using both DTAlite and SUMO
7. Compare and analyse outputs
8. Establish link to GUI front end
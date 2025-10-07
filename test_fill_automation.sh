#!/bin/bash
# Automated test for --fill flag with pre-configured answers

# Create a file with the measurements
cat > /tmp/umbra_measurements.txt << MEASUREMENTS
55.5
16
39.5
12
16
28
MEASUREMENTS

# Run with automated input
poetry run python run_cassette_system.py floorplans/Umbra_XL.png --fill < /tmp/umbra_measurements.txt

# Clean up
rm /tmp/umbra_measurements.txt

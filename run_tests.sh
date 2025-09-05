#!/bin/bash

# Start the database
echo "Starting PostgreSQL database..."
docker-compose -f databases/compose.yaml up -d postgres

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 10

# Run the tests
echo "Running PostgreSQL Arrow encoder tests..."
cargo test --features postgres -- --nocapture

# Stop the database
echo "Stopping database..."
docker-compose -f databases/compose.yaml down

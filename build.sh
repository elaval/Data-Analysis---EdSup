#!/bin/bash
set -e

echo "Starting build..."

# Try to build normally
npm run docs:build

echo "Build completed successfully"

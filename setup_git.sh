#!/bin/bash

# Git setup script for pushing to GitHub
# Author: Zijia Liu
# GitHub: m-serious

echo "========================================"
echo "Git Repository Setup for m-serious"
echo "========================================"

# Initialize git repository
echo "Initializing git repository..."
git init

# Configure git (replace with your actual email)
echo "Configuring git user..."
git config user.name "Zijia Liu"
git config user.email "your-email@example.com"  # UPDATE THIS

# Add all files
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Multi-Module Reward Model for Agent Trajectory Evaluation

- Implemented 4-dimensional scoring system for agent modules
- Added Qwen3-Embedding-0.6B as shared encoder
- Implemented two-stage training strategy
- Added Bradley-Terry preference learning
- Created comprehensive evaluation suite
- Added synthetic data generation for testing"

echo ""
echo "========================================"
echo "Repository initialized successfully!"
echo "========================================"
echo ""
echo "Next steps to push to GitHub:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: module_reward_models"
echo "   - Keep it public or private as preferred"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. After creating the empty repository, run these commands:"
echo ""
echo "   git remote add origin https://github.com/m-serious/module_reward_models.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. If you want to use SSH instead of HTTPS:"
echo "   git remote set-url origin git@github.com:m-serious/module_reward_models.git"
echo ""
echo "Your repository will be available at:"
echo "https://github.com/m-serious/module_reward_models"
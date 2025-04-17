#!/bin/bash
#
# schedule_backup.sh - Sets up a cron job to run backup.py daily at 2 AM
# 
# Usage: ./schedule_backup.sh
#
# This script:
# 1. Checks if cron is installed and running
# 2. Creates a cron job entry to run backup.py daily at 2 AM
# 3. Includes error handling for various failure scenarios
#
# Author: System Administrator
# Date: $(date +%Y-%m-%d)

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print error messages and exit
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Function to print info messages
info() {
    echo "INFO: $1"
}

# Check if the backup.py script exists
if [ ! -f "backup.py" ]; then
    error_exit "backup.py script not found in the current directory"
fi

# Make sure backup.py is executable
info "Making backup.py executable..."
chmod +x backup.py || error_exit "Failed to make backup.py executable"

# Check if cron service is installed
if ! command -v crontab &> /dev/null; then
    error_exit "crontab command not found. Please install cron service first."
fi

# Create a temporary file for the new crontab
TEMP_CRONTAB=$(mktemp) || error_exit "Failed to create temporary file"

# Export current crontab to the temporary file
crontab -l > "$TEMP_CRONTAB" 2>/dev/null || {
    # If no crontab exists, create an empty one
    touch "$TEMP_CRONTAB" || error_exit "Failed to create empty crontab"
}

# Check if the backup cron job already exists
if grep -q "backup.py" "$TEMP_CRONTAB"; then
    info "Backup cron job already exists. Updating it..."
    # Remove the existing backup job
    sed -i '/backup\.py/d' "$TEMP_CRONTAB" || error_exit "Failed to update existing cron job"
fi

# Get the full path to backup.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_SCRIPT="$SCRIPT_DIR/backup.py"

# Add the new backup job to run daily at 2 AM
echo "0 2 * * * $BACKUP_SCRIPT >> $SCRIPT_DIR/backup.log 2>&1" >> "$TEMP_CRONTAB" || error_exit "Failed to add cron job"

# Install the new crontab
crontab "$TEMP_CRONTAB" || error_exit "Failed to install new crontab"

# Clean up the temporary file
rm "$TEMP_CRONTAB" || error_exit "Failed to remove temporary file"

info "Successfully scheduled backup.py to run daily at 2 AM"
info "Backup logs will be written to $SCRIPT_DIR/backup.log"

# Make this script executable
chmod +x "$0" || error_exit "Failed to make this script executable"

exit 0


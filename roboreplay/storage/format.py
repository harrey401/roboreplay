"""RoboReplay .rrp file format constants and helpers.

The .rrp format is HDF5 with a specific structure:

    /metadata          — JSON string attr with recording info
    /schema            — JSON string attr with channel definitions
    /events            — JSON string attr with event log
    /channels/
        /<name>        — dataset per channel, shape [T, *channel_shape]
    /stats/
        /<name>        — attrs with min, max, mean, std per channel
"""

# HDF5 group/dataset paths
METADATA_ATTR = "metadata"
SCHEMA_ATTR = "schema"
EVENTS_ATTR = "events"
CHANNELS_GROUP = "channels"
STATS_GROUP = "stats"

# File extension
FILE_EXTENSION = ".rrp"

# Compression settings
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4  # compression level 1-9, 4 is good speed/ratio balance

# Chunk size for streaming writes (steps per chunk)
CHUNK_STEPS = 100

# Initial dataset allocation (grows dynamically)
INITIAL_STEPS = 1000

# Version of the format
FORMAT_VERSION = "1.0.0"

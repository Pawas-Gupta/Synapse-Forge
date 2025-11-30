# Database Connection Pool Migration Guide

## Current Status

The RFP Agent System has **two database connection methods** available:

### 1. Legacy Database Class (Currently Used)
```python
from backend.database import get_db

db = get_db()
conn = db.get_connection()
```

**Characteristics:**
- Simple sqlite3 connection
- Works for current load
- No connection pooling
- Used by all agents currently

### 2. New DatabasePool Class (Available)
```python
from backend.database import get_db_pool

pool = get_db_pool()
conn = pool.get_connection()
```

**Characteristics:**
- SQLAlchemy-based connection pooling
- Pool size: 10 connections
- Max overflow: 20 connections
- Health checks (pool_pre_ping)
- Better for high concurrency

## Why Both Exist

The connection pool was implemented as part of production enhancements, but the legacy system still works fine for current usage. Both coexist to allow gradual migration.

## When to Migrate

Migrate to connection pool when:
- ✅ Handling >50 concurrent requests
- ✅ Experiencing connection exhaustion
- ✅ Need better connection management
- ✅ Deploying to production with high load

## Migration Steps

### Step 1: Update Agent Initialization

**Before:**
```python
from backend.database import get_db

db = get_db()
db_conn = db.get_connection()
agents['sales'] = SalesAgent(db_conn)
```

**After:**
```python
from backend.database import get_db_pool

pool = get_db_pool()
db_conn = pool.get_connection()
agents['sales'] = SalesAgent(db_conn)
```

### Step 2: Update app.py

Replace in `initialize_agents()`:
```python
def initialize_agents():
    global db, agents
    
    if db is None:
        db = get_db_pool()  # Changed from get_db()
    
    if not agents:
        db_conn = db.get_connection()
        # ... rest of initialization
```

### Step 3: Test Under Load

```bash
# Run load test
python -m locust -f tests/load_test.py

# Monitor pool status
curl http://localhost:5000/api/pool-status
```

### Step 4: Monitor Metrics

Check Prometheus metrics:
- `database_pool_size`
- `database_pool_checked_out`
- `database_pool_overflow`

## Current Implementation

**Status**: ✅ Both systems working

- Legacy Database: Used by agents
- DatabasePool: Available via `/api/pool-status`
- No migration required for current load
- Pool ready when needed

## Performance Comparison

| Metric | Legacy | Pool |
|--------|--------|------|
| Concurrent Connections | 1 | 10-30 |
| Connection Reuse | No | Yes |
| Health Checks | No | Yes |
| Overhead | Low | Medium |
| Best For | Low traffic | High traffic |

## Recommendation

**For Current Deployment**: Keep legacy system
- Current load is manageable
- No connection issues observed
- Simpler debugging

**For Production Scale**: Migrate to pool
- Expected >100 concurrent users
- Multiple worker processes
- Better resource management

## Rollback Plan

If issues occur after migration:
```python
# Simply revert to:
from backend.database import get_db
db = get_db()
```

## Testing Checklist

Before migrating:
- [ ] Run full test suite
- [ ] Load test with 100+ concurrent requests
- [ ] Monitor connection metrics
- [ ] Test connection recovery
- [ ] Verify all agents work with pool
- [ ] Check memory usage

## Notes

- Both systems use SQLite (same database file)
- No data migration needed
- Can switch back and forth
- Pool adds ~10MB memory overhead
- Connection pool is thread-safe

## Last Updated
- Date: 2025-01-XX
- Status: Dual system operational
- Recommendation: Migrate when scaling to production

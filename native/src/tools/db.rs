// tools/db.rs
// Read-only SQLite retrieval over a per-task in-memory database
// Structured lookup, free SQL with authorizer, row cap, and time cap

use super::{string, Registry, World};
use anyhow::{bail, Context, Result};
use rusqlite::{
    hooks::{AuthAction, AuthContext, Authorization},
    params_from_iter,
    types::{Value as Sql, ValueRef},
    Connection, ErrorCode,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

const ROW_CAP: usize = 20;
const QUERY_TIME: Duration = Duration::from_millis(500);

pub struct Db {
    conn: Connection,
    schema: BTreeMap<String, Vec<(String, String)>>,
}
fn plain(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
}
fn cell(v: ValueRef<'_>) -> Value {
    match v {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => json!(i),
        ValueRef::Real(f) => json!(f),
        ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
        ValueRef::Blob(b) => json!(format!("<blob {} bytes>", b.len())),
    }
}
fn bind(v: &Value) -> Result<Sql> {
    Ok(match v {
        Value::String(s) => Sql::Text(s.clone()),
        Value::Bool(b) => Sql::Integer(i64::from(*b)),
        Value::Number(n) => n
            .as_i64()
            .map(Sql::Integer)
            .or_else(|| n.as_f64().map(Sql::Real))
            .context("number out of range")?,
        _ => bail!("where values must be strings, numbers, or booleans"),
    })
}
impl Db {
    /// Seed an in-memory database, then lock it to reads.
    pub fn open(seed: &str) -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(seed).context("database seed failed")?;
        let mut schema = BTreeMap::new();
        let names = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")?
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for name in names {
            if !plain(&name) {
                bail!("table name {name} must be letters, digits, or underscores");
            }
            let columns = conn
                .prepare(&format!("PRAGMA table_info(\"{name}\")"))?
                .query_map([], |r| Ok((r.get::<_, String>(1)?, r.get::<_, String>(2)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            if columns.iter().any(|(c, _)| !plain(c)) {
                bail!("column names in {name} must be letters, digits, or underscores");
            }
            schema.insert(name, columns);
        }
        conn.execute_batch("PRAGMA query_only=ON")?;
        conn.authorizer(Some(|ctx: AuthContext<'_>| match ctx.action {
            AuthAction::Select
            | AuthAction::Read { .. }
            | AuthAction::Function { .. }
            | AuthAction::Recursive => Authorization::Allow,
            _ => Authorization::Deny,
        }));
        Ok(Self { conn, schema })
    }
    /// Rows from one read-only statement, capped by count and time.
    fn run(&self, sql: &str, params: &[Sql]) -> Result<Value> {
        let started = Instant::now();
        self.conn
            .progress_handler(1000, Some(move || started.elapsed() > QUERY_TIME));
        let result = (|| -> rusqlite::Result<Value> {
            let mut stmt = self.conn.prepare(sql)?;
            if !stmt.readonly() {
                return Err(rusqlite::Error::InvalidQuery);
            }
            let columns = stmt
                .column_names()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let mut rows = stmt.query(params_from_iter(params))?;
            let mut out = Vec::new();
            let mut truncated = false;
            while let Some(row) = rows.next()? {
                if out.len() == ROW_CAP {
                    truncated = true;
                    break;
                }
                out.push(
                    (0..columns.len())
                        .map(|i| row.get_ref(i).map(cell))
                        .collect::<rusqlite::Result<Vec<_>>>()?,
                );
            }
            Ok(json!({"columns":columns,"rows":out,"truncated":truncated}))
        })();
        self.conn.progress_handler(0, None::<fn() -> bool>);
        result.map_err(|e| match e {
            rusqlite::Error::SqliteFailure(f, _) if f.code == ErrorCode::OperationInterrupted => {
                anyhow::anyhow!("query exceeded the time limit")
            }
            rusqlite::Error::InvalidQuery => anyhow::anyhow!("only read-only queries are allowed"),
            e => anyhow::anyhow!("{e}"),
        })
    }
    fn lookup(&self, a: &Value) -> Result<Value> {
        let table = string(a, "table")?;
        let columns = self
            .schema
            .get(table)
            .with_context(|| format!("unknown table {table}"))?;
        let known = |c: &str| columns.iter().any(|(n, _)| n == c);
        let selected = match a["columns"].as_array().filter(|v| !v.is_empty()) {
            Some(v) => v
                .iter()
                .map(|c| {
                    c.as_str()
                        .filter(|c| known(c))
                        .map(String::from)
                        .with_context(|| format!("unknown column {c}"))
                })
                .collect::<Result<Vec<_>>>()?,
            None => columns.iter().map(|(n, _)| n.clone()).collect(),
        };
        let mut clauses = Vec::new();
        let mut params = Vec::new();
        for (k, v) in a["where"].as_object().into_iter().flatten() {
            if !known(k) {
                bail!("unknown column {k}");
            }
            clauses.push(format!("\"{k}\" = ?"));
            params.push(bind(v)?);
        }
        let limit = a["limit"].as_u64().unwrap_or(10).clamp(1, ROW_CAP as u64);
        let filter = if clauses.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", clauses.join(" AND "))
        };
        let select = selected
            .iter()
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let mut out = self.run(
            &format!(
                "SELECT {select} FROM \"{table}\"{filter} LIMIT {}",
                limit + 1
            ),
            &params,
        )?;
        let capped = out["truncated"] == true;
        let rows = out["rows"].as_array_mut().context("rows missing")?;
        let truncated = capped || rows.len() as u64 > limit;
        rows.truncate(limit as usize);
        out["truncated"] = json!(truncated);
        Ok(out)
    }
}
/// Fail early when a seed does not load.
pub fn check_seed(seed: &str) -> Result<()> {
    Db::open(seed).map(|_| ())
}
fn db(w: &World) -> Result<&Db> {
    w.db.as_ref().context("this task has no database")
}
fn db_schema(w: &mut World, _: &Value) -> Result<Value> {
    let tables = db(w)?
        .schema
        .iter()
        .map(|(name, cols)| {
            json!({"name":name,"columns":cols.iter().map(|(n, t)| json!({"name":n,"type":t})).collect::<Vec<_>>()})
        })
        .collect::<Vec<_>>();
    Ok(json!({"tables":tables}))
}
fn db_lookup(w: &mut World, a: &Value) -> Result<Value> {
    db(w)?.lookup(a)
}
fn db_query(w: &mut World, a: &Value) -> Result<Value> {
    let sql = string(a, "sql")?.trim().trim_end_matches(';');
    if sql.contains(';') {
        bail!("send one statement without semicolons");
    }
    db(w)?.run(sql, &[])
}
pub fn register(r: &mut Registry) {
    r.insert("db_schema", db_schema);
    r.insert("db_lookup", db_lookup);
    r.insert("db_query", db_query);
}

#[cfg(test)]
mod tests {
    use crate::tools::test_support::host;
    use serde_json::json;

    const SEED: &str =
        "CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, stock INTEGER, price REAL);
        INSERT INTO products VALUES (1,'bolt',5,0.25),(2,'nut',12,0.1),(3,'washer',40,0.05);";
    fn seeded() -> (crate::tools::Builtins, tempfile::TempDir) {
        host(json!({"db":{"sql":SEED}}))
    }

    #[test]
    fn schema_lists_tables_and_column_types() {
        let (mut h, _dir) = seeded();
        let s = h.call("db_schema", json!({})).unwrap();
        assert_eq!(s["tables"][0]["name"], "products");
        assert_eq!(
            s["tables"][0]["columns"][2],
            json!({"name":"stock","type":"INTEGER"})
        );
    }
    #[test]
    fn lookup_filters_selects_columns_and_coerces_text_numbers() {
        let (mut h, _dir) = seeded();
        let r = h
            .call(
                "db_lookup",
                json!({"table":"products","where":{"stock":12},"columns":["name"]}),
            )
            .unwrap();
        assert_eq!(r["rows"], json!([["nut"]]));
        let text = h
            .call(
                "db_lookup",
                json!({"table":"products","where":{"stock":"12"}}),
            )
            .unwrap();
        assert_eq!(text["rows"].as_array().unwrap().len(), 1);
        let all = h
            .call("db_lookup", json!({"table":"products","limit":2}))
            .unwrap();
        assert_eq!(
            (
                all["rows"].as_array().unwrap().len(),
                all["truncated"].clone()
            ),
            (2, json!(true))
        );
        assert!(h.call("db_lookup", json!({"table":"nope"})).is_err());
        assert!(h
            .call("db_lookup", json!({"table":"products","where":{"zzz":1}}))
            .is_err());
        assert!(h
            .call(
                "db_lookup",
                json!({"table":"products","columns":["a\" FROM x --"]})
            )
            .is_err());
    }
    #[test]
    fn query_runs_selects_and_rejects_writes_and_multiple_statements() {
        let (mut h, _dir) = seeded();
        let r = h
            .call("db_query", json!({"sql":"SELECT name, stock * price AS value FROM products WHERE stock > 10 ORDER BY id;"}))
            .unwrap();
        assert_eq!(r["columns"], json!(["name", "value"]));
        assert_eq!(r["rows"].as_array().unwrap().len(), 2);
        for sql in [
            "INSERT INTO products VALUES (9,'x',1,1)",
            "DELETE FROM products",
            "DROP TABLE products",
            "UPDATE products SET stock = 0",
            "ATTACH DATABASE ':memory:' AS other",
            "PRAGMA query_only = OFF",
            "SELECT 1; DELETE FROM products",
        ] {
            assert!(h.call("db_query", json!({"sql":sql})).is_err(), "{sql}");
        }
        let n = h
            .call("db_query", json!({"sql":"SELECT count(*) FROM products"}))
            .unwrap();
        assert_eq!(n["rows"], json!([[3]]));
    }
    #[test]
    fn runaway_queries_are_cut_off_and_rows_are_capped() {
        let (mut h, _dir) = seeded();
        let slow = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT count(*) FROM n";
        let started = std::time::Instant::now();
        let err = h
            .call("db_query", json!({"sql":slow}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("time limit"), "{err}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        let many = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 100) SELECT x FROM n";
        let r = h.call("db_query", json!({"sql":many})).unwrap();
        assert_eq!(
            (r["rows"].as_array().unwrap().len(), r["truncated"].clone()),
            (20, json!(true))
        );
    }
    #[test]
    fn tasks_without_a_seed_report_no_database() {
        let (mut h, _dir) = host(json!({}));
        assert!(h.call("db_schema", json!({})).is_err());
    }
}

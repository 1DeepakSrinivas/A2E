CREATE TABLE sensors(id INTEGER PRIMARY KEY, name TEXT, location TEXT, unit TEXT);
INSERT INTO sensors VALUES
 (1,'cooler','lab','C'),
 (2,'heater','lab','C'),
 (3,'intake','roof','C'),
 (4,'tank','basement','L');
CREATE TABLE readings(sensor_id INTEGER, taken_on TEXT, value REAL);
INSERT INTO readings VALUES
 (1,'2026-03-01',31.5),(1,'2026-03-02',28.0),(1,'2026-03-03',35.2),
 (2,'2026-03-01',60.0),(2,'2026-03-02',62.5),
 (3,'2026-03-01',12.0),(3,'2026-03-02',9.5),(3,'2026-03-03',14.0),
 (4,'2026-03-01',420.0),(4,'2026-03-02',390.0);

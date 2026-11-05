CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER);
INSERT INTO products VALUES
 (1,'bolt','fasteners',0.25,5),
 (2,'nut','fasteners',0.10,12),
 (3,'washer','fasteners',0.05,40),
 (4,'drill','tools',49.90,3),
 (5,'hammer','tools',15.50,8),
 (6,'wrench','tools',12.00,0),
 (7,'glue','supplies',4.20,25),
 (8,'tape','supplies',2.75,60),
 (9,'sandpaper','supplies',1.10,2),
 (10,'paint','supplies',18.00,7);
CREATE TABLE orders(id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER, status TEXT, ordered_on TEXT);
INSERT INTO orders VALUES
 (1,1,100,'shipped','2026-02-01'),
 (2,4,2,'pending','2026-02-10'),
 (3,6,10,'pending','2026-02-12'),
 (4,9,50,'shipped','2026-02-14'),
 (5,2,200,'delivered','2026-02-15'),
 (6,5,4,'delivered','2026-02-20'),
 (7,10,6,'pending','2026-02-25'),
 (8,3,500,'shipped','2026-02-27');

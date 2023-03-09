CREATE TABLE Payment (
  id INT PRIMARY KEY AUTO_INCREMENT,
  payment_type VARCHAR(50) NOT NULL,
  payment_status_id INT,
  payment_time DATETIME,
  FOREIGN KEY (payment_status_id) REFERENCES PaymentStatus(id)
);

CREATE TABLE ShippingAddress (
  id INT PRIMARY KEY AUTO_INCREMENT,
  recipient_name VARCHAR(50) NOT NULL,
 recipient_phone VARCHAR(15) NOT NULL,
  recipient_address VARCHAR(100) NOT NULL
);

CREATE TABLE OrderStatus (
  id INT PRIMARY KEY AUTO_INCREMENT,
  status VARCHAR(50) NOT NULL
);

CREATE TABLE Invoice (
  id INT PRIMARY KEY AUTO_INCREMENT,
  invoice_type VARCHAR(50) NOT NULL,
  invoice_header VARCHAR(50),
  invoice_amount DECIMAL(10, 2),
  invoice_time DATETIME
);

CREATE TABLE PaymentStatus (
  id INT PRIMARY KEY AUTO_INCREMENT,
  status VARCHAR(50) NOT NULL
);  




CREATE TABLE `Order` (
  id INT PRIMARY KEY AUTO_INCREMENT,
  order_no VARCHAR(50) NOT NULL,
  customer_id INT NOT NULL,
  payment_id INT,
  shipping_address_id INT,
  order_status_id INT,
  invoice_id INT,
  created_at DATETIME,
  updated_at DATETIME,
  FOREIGN KEY (customer_id) REFERENCES Customer(id),
  FOREIGN KEY (payment_id) REFERENCES Payment(id),
  FOREIGN KEY (shipping_address_id) REFERENCES ShippingAddress(id),
  FOREIGN KEY (order_status_id) REFERENCES OrderStatus(id),
  FOREIGN KEY (invoice_id) REFERENCES Invoice(id)
);

CREATE TABLE OrderItem (
  id INT PRIMARY KEY AUTO_INCREMENT,
  order_id INT NOT NULL,
  item_name VARCHAR(50) NOT NULL,
  item_price DECIMAL(10, 2) NOT NULL,
  item_quantity INT NOT NULL,
  FOREIGN KEY (order_id) REFERENCES `Order`(id)
);


INSERT INTO Customer (name, phone, address) VALUES 
('张三', '13888888888', '上海市浦东新区'),
('李四', '13999999999', '北京市海淀区'),
('王五', '13666666666', '广州市天河区');

INSERT INTO PaymentStatus (status) VALUES 
('已付款'),
('未付款');

INSERT INTO Payment (payment_type, payment_status_id, payment_time) VALUES 
('支付宝', 1, '2022-06-01 10:30:00'),
('微信', 1, '2022-06-02 08:45:00'),
('银行卡', 2, NULL);

INSERT INTO ShippingAddress (recipient_name, recipient_phone, recipient_address) VALUES 
('张三', '13888888888', '上海市浦东新区东方路800号'),
('李四', '13999999999', '北京市海淀区中关村大街123号'),
('王五', '13666666666', '广州市天河区太阳城2期');

INSERT INTO OrderStatus (status) VALUES 
('待付款'),
('已付款'),
('待发货'),
('已完成');

INSERT INTO Invoice (invoice_type, invoice_header, invoice_amount, invoice_time) VALUES 
('增值税普通发票', 'ABC有限公司', 998.00, '2022-06-02 13:22:00'),
('增值税专用发票', NULL, 598.00, '2022-06-01 15:55:00'),
('不开发票', NULL, 249.00, NULL);

INSERT INTO `Order` (order_no, customer_id, payment_id, shipping_address_id, order_status_id, invoice_id, created_at, updated_at) VALUES 
('20220601001', 1, 1, 1, 2, 2, '2022-06-01 09:00:00', '2022-06-02 10:00:00'),
('20220601002', 2, 3, 2, 1, NULL, '2022-06-01 14:00:00', '2022-06-02 16:00:00'),
('20220602001', 3, 2, 3, 3, 1, '2022-06-02 08:00:00', NULL);

INSERT INTO OrderItem (order_id, item_name, item_price, item_quantity) VALUES 
(1, '商品A', 499.00, 1),
(1, '商品B', 299.00, 1),
(1, '商品C', 200.00, 1),
(2, '商品D', 129.00, 2),
(2, '商品E', 120.00, 1),
(3, '商品F', 98.00, 2),
(3, '商品G', 49.00, 3);

SELECT Customer.phone, COUNT(*) AS order_count
FROM Customer
INNER JOIN `Order` ON Customer.id = `Order`.customer_id
GROUP BY Customer.phone;


SELECT MONTH(created_at) AS month, COUNT(*) AS order_count
FROM `Order`
GROUP BY MONTH(created_at);

SELECT DATE(created_at), COUNT(id) 
FROM `Order` 
GROUP BY DATE(created_at);


SELECT DATE_FORMAT(created_at,'%Y%m') AS month, COUNT(DISTINCT phone) AS phone_count
FROM `Order`
JOIN Customer ON Order.customer_id = Customer.id
GROUP BY month;


SELECT DATE_FORMAT(o.created_at, '%Y-%m') AS month, COUNT(DISTINCT c.phone) AS phone_count 
FROM Customer c 
JOIN Order o ON c.id = o.customer_id 
GROUP BY month 
ORDER BY month ASC; 

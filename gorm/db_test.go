//
// db_test.go
// Copyright (C) 2025 n3xtchen <echenwen@gmail.com>
//
// Distributed under terms of the GPL-2.0 license.
//

package db

import (
	"testing"
)

func TestDB(t *testing.T) {
	InitDB()

	db = NewDBConn()

	// db.Create(&Product{Code: "D42", Price: 100})

	// // Read
	var product Product
	db.First(&product, 1) // find product with integer primary key
	// db.First(&product, "code = ?", "D42") // find product with code D42

	// // Update - update product's price to 200
	// db.Model(&product).Update("Price", 200)
	// // Update - update multiple fields
	// db.Model(&product).Updates(Product{Price: 200, Code: "F42"}) // non-zero fields
	// db.Model(&product).Updates(map[string]interface{}{"Price": 200, "Code": "F42"})

	// // Delete - delete product
	// db.Delete(&product, 1)
}

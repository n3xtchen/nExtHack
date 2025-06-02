//
// db_test.go
// Copyright (C) 2025 n3xtchen <echenwen@gmail.com>
//
// Distributed under terms of the GPL-2.0 license.
//

package db

import (
	"fmt"
	"testing"
)

func TestDB(t *testing.T) {
	// InitDB()

	xdb := NewDBConn()

	// xdb.Create(&Category{Name: "XX"})
	// xdb.Create(&Product{Code: "D42", Price: 100, CategoryID: 1})

	// Read
	// var product Product
	// xdb.First(&product, 8) // find product with integer primary key
	// fmt.Println(product.Price)

	// xdb.First(&product, "code = ?", "D42") // find product with code D42
	// fmt.Print(product)

	// Update - update product's price to 200
	// xdb.Model(&product).Update("Price", 200)
	// fmt.Println(product.Price)
	// Update - update multiple fields
	// xdb.Model(&product).Updates(Product{Price: 200, Code: "F42"}) // non-zero fields
	// fmt.Println(product.Code)
	// xdb.Model(&product).Updates(map[string]interface{}{"Price": 200, "Code": "F42"})
	// fmt.Println(product)

	// // Delete - delete product
	// xdb.Unscoped().Delete(&product, 1)
	var count int64
	xdb.Model(&Product{}).Count(&count)
	fmt.Println("商品表记录数:", count)

	var products []Product

	result := xdb.Preload("Category").Find(&products)
	fmt.Println(result.RowsAffected)
	fmt.Println(result.Error)
	fmt.Println(products)

	fmt.Println("for loop")
	for i := 0; i < len(products); i++ {
		fmt.Println(products[i].Code)
		fmt.Println(products[i].CategoryID)
		fmt.Println(products[i].Category.Name)
	}

	fmt.Println("for range")
	for _, row := range products {
		fmt.Println(row)
	}

}

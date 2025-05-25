//
// db.go
// Copyright (C) 2025 n3xtchen <echenwen@gmail.com>
//
// Distributed under terms of the GPL-2.0 license.
//

package db

import (
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

type Product struct {
	gorm.Model
	Code  string
	Price uint
	Rate  uint
}

var db *gorm.DB

func InitDB() {
	db, err := gorm.Open(sqlite.Open("test.db"), &gorm.Config{})
	if err != nil {
		panic("failed to connect database")
	}

	// 迁移 schema
	db.AutoMigrate(&Product{})
}

func NewDBConn() *gorm.DB {
	mydb := db
	return mydb
}

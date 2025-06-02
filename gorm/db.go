//
// db.go
// Copyright (C) 2025 n3xtchen <echenwen@gmail.com>
//
// Distributed under terms of the GPL-2.0 license.
//

package db

import (
	"sync"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

type Category struct {
	gorm.Model
	Name     string
	Products []Product
}

type Product struct {
	gorm.Model
	Code       string
	Price      uint
	Rate       uint
	CategoryID uint
	Category   Category
}

var _db *gorm.DB
var once sync.Once

func InitDB() {
	db, err := gorm.Open(sqlite.Open("test.db"), &gorm.Config{})
	if err != nil {
		panic("failed to connect database")
	}

	// 迁移 schema
	db.AutoMigrate(&Product{}, &Category{})

	_db = db
}

func NewDBConn() *gorm.DB {

	once.Do(func() {
		InitDB()
	})

	if _db == nil {
		panic("singleton instance is nil after initialization")
	}

	return _db
}

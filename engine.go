package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	chroma "github.com/amikos-tech/chroma-go"
)

func main() {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	downloadsPath := filepath.Join(homeDir, "Downloads")

	fmt.Println("Scanning PDFs in:", downloadsPath)

	var pdfFiles []string

	err = filepath.WalkDir(downloadsPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".pdf") {
			pdfFiles = append(pdfFiles, path)
		}
		return nil
	})
	if err != nil {
		panic(err)
	}

	if len(pdfFiles) == 0 {
		fmt.Println("No PDFs found in Downloads folder.")
		return
	}

	fmt.Println("Found PDFs:")
	for _, pdf := range pdfFiles {
		fmt.Println("-", pdf)
	}

	c, err := chroma.NewClient(chroma.WithBasePath("http://localhost:8000"))
	if err != nil {
		panic(err)
	}

	// No embedding function needed for querying existing collection
	collection, err := c.GetCollection(
		context.Background(),
		"my_pdfs",
		nil,
	)
	if err != nil {
		panic(err)
	}

	queryText := "concurrency in Go"

	results, err := collection.Query(
		context.Background(),
		[]string{queryText},
		5,
		nil,
		nil,
		nil,
	)
	if err != nil {
		panic(err)
	}

	for _, docs := range results.Documents {
		for _, doc := range docs {
			fmt.Println("Found relevant concept in:", doc)
		}
	}
}
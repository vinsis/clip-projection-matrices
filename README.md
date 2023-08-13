# clip-projection-matrices

## Dataset
- Source: http://database.mmsp-kn.de/koniq-10k-database.html
    - [Small dataset](http://datasets.vqa.mmsp-kn.de/archives/koniq10k_512x384.zip) of ~10,000 images

## Repo structure
```sh
.
├── LICENSE
├── README.md
├── assets
│   ├── bottom_10_images.png
│   ├── image_vectors.pt # (added to .gitginore)
│   └── top_10_images.png
├── images
│   └── dataset # (added to .gitignore)
└── src
    ├── main.py
    ├── utils.py
    └── visualizing_results.ipynb
```

### Top 10 images based on the prompts
![](assets/top_10_images.png)

### Bottom 10 images based on the prompts
![](assets/bottom_10_images.png)

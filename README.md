# AI/ML Club - Image Denoising with Autoencoders

**Academic Year:** 2025-2026 Fall Semester  
**Project Duration:** September 2025 - December 2025

## Project Description

This project explores the problem of **image denoising**—removing unwanted noise from images while preserving essential details. Using deep learning autoencoders, we begin with baseline convolutional architectures and progressively move toward advanced designs inspired by modern backbones like EfficientNet. The goal is to build a custom denoiser, compare it with classical approaches, and evaluate it against real-world denoising algorithms.

**Key Objectives:**

-   Understand the fundamentals of autoencoders for image denoising
-   Build baseline models on small datasets (e.g., CIFAR10)
-   Scale to larger datasets such as Natural Images with Synthetic Noise
-   Experiment with advanced architectures and compare against state-of-the-art methods

## Lead Contact Information

**Project Lead:** Aditya Hegde  
📧 Email: adityahegde712@gmail.com / aditya.hegde@sjsu.edu <br>
💼 LinkedIn: [linkedin.com/in/aditya-hegde712](https://www.linkedin.com/in/aditya-hegde712/)  
📱 Phone: –  
🏢 Office Hours: -

**Faculty Advisor:** None

## Contributors

_For detailed member information including LinkedIn profiles and Discord handles, see [docs/members.csv](docs/members.csv)_

| Name                          | Role          | Email                      | GitHub                                               |
| ----------------------------- | ------------- | -------------------------- | ---------------------------------------------------- |
| Aditya Hegde                  | Project Lead  | adityahegde712@gmail.com   | [@AdityaHegde712](https://github.com/AdityaHegde712) |
| Leonardo Flores Gonzalez(Leo) | Sub-Team Lead | leof7812@gmail.com         | [@leo7812](https://github.com/leo7812)               |
| Abhishek Darji                | Sub-Team Lead | abhishekdarji653@gmail.com | [@Darji23](https://github.com/Darji23)               |

_(See members.csv for full roster)_

## Project Kanban Board

**🔗 Public Board:** –

## Repository Structure

Note: Code is currently private except for a tutorial notebook in `common/` in case anyone else would like to get started. We will release the code once we have our first set of working models!

```
denoise-ae/
├─ README.md
├─ environment.yml           # environment setup file
├─ .gitignore
├─ common/                   # Any common use or data prep scripts/notebooks if required
├─ docs/                     # Member and project info
│  ├─ members.csv            # CSV file with member info
│  ├─ info.json              # Basic project info
│  ├─ pitch_slides.pdf       # Slides for initial project pitch
│  └─ thumbnail.webp         # Project thumbnail
├─ eval/                     # Store the evaluation split here for benchmarks and comparisons
├─ teams/
│  ├─ team-A/
│  │  ├─ model/              # team’s specific model
│  │  ├─ experiments/        # configs + notes
│  │  └─ results/            # small artifacts (plots/metrics JSON)
│  │  ├─ scripts/            # a folder for team-wise scripts
│  ├─ team-B/
└─ └─ team-C/
```

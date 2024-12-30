# Quantum Education Modules Documentation

This folder contains the setup for hosting the **Read the Docs** [website](https://csci4961-labs-projects.readthedocs.io/en/latest/) for this GitHub.

## Structure

The documentation is organized as follows:

- **Home**: Homepage with links to documentation and modules
- **User Documentation**: Overview of project, roadmap, usage, and how to contribute.
- **Introductory Modules**: Section for all the introductory modules in [../Modules/Concepts](../Modules/Concepts).
- **Intermediate Modules**: Section for all intermediate modules in [../Modules/Algorithms](../Modules/Algorithms).
- **Advanced Modules**: Section for all advanced modules in [../Modules/Applications](../Modules/Applications).

Alternatively, you can clone this repository and build the documentation locally, using conda:

```bash
# Clone the repository
git clone https://github.com/YangletLiu/CSCI4961_labs_projects.git

# Navigate to the docs folder
cd CSCI4961_labs_projects/docs

# Install dependencies
conda env create --name test --file environment.yml

# Activate environment
conda activate test

# Build the documentation
sphinx-build -b html source/ _build/
```

The output HTML files will be located in the `_build/` directory.

## Contributing

We welcome contributions from the community to enhance and expand these educational resources. Please read our [Contributing Guide](https://csci4961-labs-projects.readthedocs.io/en/latest/user/contribute.html#) for more details on how to get involved.

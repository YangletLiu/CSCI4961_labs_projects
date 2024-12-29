=================
How to Contribute
=================

Contributing to our open-source quantum education modules will strengthen your understanding of quantum mechanics and build valuable technical skills with Qiskit or real quantum hardware. 

You will gain hands-on experience, expand your professional network, and boost your credibility as an open-source contributor—opening doors to career opportunities in quantum computing.

Here is how you can get involved!

.. contents:: Table of Contents
   :local:

Suggested Contributions
=======================

Feedback on Existing Modules
----------------------------
You can provide feedback or suggest improvements to existing modules by submitting an issue on our `GitHub <https://github.com/YangletLiu/CSCI4961_labs_projects>`_. Examples of feedback can include:

- Adding examples
- Correcting errors
- Improving explanations

Feedback is critical in helping us iterate on these modules and refine the quality of our content.

Link Existing Modules to this Site
----------------------------------
To keep this documentation up-to-date, you can contribute by linking existing modules from the `GitHub <https://github.com/YangletLiu/CSCI4961_labs_projects>`_ to this site. This will make it easier for users to navigate through the modules and find the information they need.

More information is available here: `Linking Existing Modules`_.

Create New Content
------------------
Contributors are also encouraged to suggest quantum topics or create new modules based on our :doc:`Roadmap <roadmap>`, existing suggestions, or your own ideas.

You can propose quantum topics by submitting an issue on our `GitHub <https://github.com/YangletLiu/CSCI4961_labs_projects>`_. This helps spark interest and engage students who may be curious about the same concept area.

You can create new modules by developing a Jupyter notebook on a topic of your choice, following the structure of existing modules. This process deepens your understanding of the topic and allows you to share your knowledge with others. 

To submit new modules, you will need to `Contribute Directly via Pull Requests`_.


Help with Formatting/Structure
------------------------------
For quantum computing newcomers interested in contributing, you can help by improving the structure, design, or flow of existing content to make it more readable and engaging. 
This gives you the opportunity to learn about quantum computing while contributing to the community.

To submit any changes in the formatting, structure, or design of the modules, you will need to `Contribute Directly via Pull Requests`_.  

Linking Existing Modules
========================

Linking
-------
To link existing modules to this site, we use nbsphinx-link. You will need to work in the docs/source/modules folder in the repository.

If there is an existing index.rst file in the module folder, you can add the module to the toctree. If there is no index.rst file, you will need to create one.
The items of the toctree are linked to the relative paths of the file you want to link.

.. parsed-literal::
    .. toctree::
        :maxdepth: 1
        :caption: Example (Page Header)

        existingModule1 (nblink file)
        existingModule2/index (rst file)
        example (nblink file)

For Jupyter Notebooks you want to link, you create a .nblink file with the following content:

.. parsed-literal::

    {
        "path": "relative/path/to/notebook"
    }

If there are images accompanying the notebook, you copy the image folder and place it in the same directory as the .nblink.

.. parsed-literal::
    example_images
    example1_images
    example.nblink
    example1.nblink

Testing
-------

Make sure you are in the docs/ directory. To test locally, you will need to install conda and create a new environment:

.. parsed-literal::
    conda env create --name test --file environment.yml

Activate the environment:

.. parsed-literal::
    conda activate test

Then, you can build the documentation:

.. parsed-literal::
    sphinx-build -b html source/ _build/

Click on the index.html file in the _build/ directory to verify the changes.

Finally, you will need to `Contribute Directly via Pull Requests`_.


Contribute Directly via Pull Requests
=====================================
As an alternative to submitting issues, you can contribute directly by creating a pull request (PR). 

You will need to fork the repository:

.. parsed-literal::

    git clone https://github.com/YangletLiu/CSCI4961_labs_projects.git

In this forked repository, you can make your changes. To check all your unstaged changes, use:

.. parsed-literal::

    git status

Then, add all the changes you want to be pushed and then commit:

.. parsed-literal::

    git add "../Modules/Concepts/Example/Example.ipynb"
    git commit -m "Example text: Added new module"

After committing, push your changes to your forked repository:

.. parsed-literal::

    git push

Finally, create a pull request on the original repository. Your changes will then be reviewed and merged.
# Leonardo.ai Challenge Stage

<p align="center">
  <img src="./assets/introduction_image.png" width="720"/>
</p>

This project aims to implement a solution that can compute a similarity metric for image-text pairs. We then leverage this solution to enrich a given dataset, ultimately giving a measure of how similar the image is to the provided caption.

In this README we'll discuss some of design considerations along with providing a brief discussion around the scalability and performance of the solution along with discussing some of the potential practical applications of such a solution.

# Table of Contents

- [Project Structure](#project-structure)
- [Challenge 1](#challenge-1)
  - [Installation](#installation)
  - [Design Considerations](#design-considerations)
- [Challenge 2](#challenge-2)
  - [Time and Memory Footprint](#time-and-memory-footprint)
  - [Scaling to 100 Million](#scaling-to-100-million)
- [Challenge 3](#challenge-3)

## Project Structure

In this project we provide a couple main points of interest.

👉 First is the [main.ipynb](main.ipynb) notebook. This notebook is intended to walk the team at Leonardo.ai through my solution and thought process. So this is a good place to start for challenge 1. Again, a notebook isn't something we'd want to productionise, it's purely for readability benefits for this challenge.

👉 Second is the [image_text_similarity_calculator.py](src/image_text_similarity_calculator.py) python file, which is the main class that implements our similarity calculation solution.

## Challenge 1

> Develop some code to compute the similarity metric for each image-text pair and save it in an additional column in the given csv file

For challenge 1 we go into a fair amount of detail in the [main.ipynb](main.ipynb) notebook. This notebook walks through my thought process attempting to provide a solution to this problem along with delving into some of the conceptual aspects of this type of problem. 

### Installation

I'm going to assume you know your way around Python 🐍, such that you know how to manage different versions of python, create and activate virtual environments, install dependencies and run a Jupyter notebook.

Here's what you need:

- Python version can be found [.python-version](.python-version)
- All dependencies can be found [requirements.txt](requirements.txt)

### Design Considerations

As mentioned in the challenge document this is an intentionally very open task. This is great, but it's also really difficult to know when to stop! Hopefully in this section clarifies any pieces I've left out.

There's also a fair amount of context and reasoning in the [main.ipynb](main.ipynb) notebook.

#### Class Design

The main abstraction for our solution is the `ImageTextSimilarityCalculator` class. This can be found in the [image_text_similarity_calculator.py](src/image_text_similarity_calculator.py) file. This class is responsible for calculating the similarity scores for a given set of inputs.

This class is intentionally designed with composition in mind - focusing on the ability to vary what underlying zero-shot image classification models and distance calculation concrete components are used. This flexibility ensures our solution remains adaptable and future-proof.

> This sort of dependency inject / composition pattern greatly improves the testability of the class as-well!

From here we see a consistent pattern used for both the [model](src/model) and [distance](src/distance) modules. These modules both contain a base class (E.g. [distance/distance_calculator.py](src/distance/distance_calculator.py)) that the concrete implementations ([distance/cosine_distance_calculator.py](src/distance/cosine_distance_calculator.py)) inherit. This supports our above composibility as we enforce a common interface between concrete implementation.

#### Development Process

Outside of the context of this small toy technical example the development posture and process would look a lot more mature than what is provided here. This ensures we as team develop and adhere to best practices and standards, enables maintainability, scalability and promotes agility.

A few things that we'd consider are:

- **Version control**: git is my and my team's "daily driver", we follow a strict [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) workflow (master and feature branches). This workflow is also imperative to how we build and deploy our software.
- **Code Quality and Linting**: Another big one for myself and the team, we follow a set of internal standards that are enforced by development tooling (black, flake8 etc) and code review processes. This ensures we have a consistent coding style and helps to catch errors early.
- **Automated Testing**: Writing automated tests for our code is another imperative development process we adhere to, using frameworks like `pytest` help to catch bugs early. These tests also allow us to have confidence in the code we deploy, and ultimately get run and validated by our CI / CD systems.
- **Consistent Development Environments**: One thing we've been loving lately is developing inside of [development container](https://code.visualstudio.com/docs/devcontainers/containers). This drastically decreased the barrier for our developers to get their environment setup correctly and is something we're seeing more commonly across the open source community as-well.

#### Deployment Process

Another area I also want briefly touch on is deployment. Spending a fair amount of years in the cloud operation space a lot of my time was focusing on helping our developers deploy their solutions reliably at scale. A couple things that can really help in this space are:

- **Containerisation**: Tools like docker are great at enabling consistent and reliable deployments. This helps ensure our software is portable across environments I.e. deploying to a development environment should be as production like, helping reduce inconsistencies. There's also a bunch of other rabbit-holes to dive down into with containers and their usefulness ([hermetric builds](https://bazel.build/basics/hermeticity)).
- **Automated Deployment**: Deploying our software with automation helps remove manual requirements from the deployment loop. Manual interventions are were a lot of deployment issues can come from! Checkout [a DevOps Cautionary Tale](https://dougseven.com/2014/04/17/knightmare-a-devops-cautionary-tale/) which highlights this perfectly!
- **Infrastructure as Code**: Another invaluable process is defining our infrastructure as code. This helps with consistent deployments, disaster recovery, testability etc!

## Challenge 2

> Briefly discuss the time and memory footprint of computing the similarity metric. How would you optimize your implementation If you were to do this on a scale of around ~100 million image-text pairs?

### Time and Memory Footprint

Computing the similarity of our image-text pairs is relatively compute intensive. For this use-case we're only considering the impact inference has on this footprint.

👉 First, loading and instantiating a large transformer model directly impacts the memory footprint.

👉 Second, when inferring each forward pass through the model has a time cost associated with it. Usually the larger the network the more calculations that are required and ultimately the more time is required.

👉 Third, there's also preprocessing and postprocessing elements that also need to be considered and could directly have an impact on the memory and time performance of our solution.

### Scaling to 100 Million

How we scale to 100 million image-text pairs depends on our requirements. For example, we could be providing an online model that is expected to make 100 million image-text pair similarity calculations over a year - or we could be implementing a system the we want to batch score 100 million image-text pairs to enhance a training dataset for another model.

Let's assuming we're more in this second batch use-case realm.

Naturally to scale our solution to support such large amounts of data we need to balance the above time and memory footprints, while also ensure a solid operational posture - not to mention all of the cost and security considerations. Things we'd want to consider:

#### Parallel Processing and Distributed Compute

It wouldn't be practical to process 1 image-text pair at a time, instead we'd want to process a bunch our image-text pairs in parallel and distribute the work across multiple compute instances.

👉 Ensure our implementation supports batch processing of image-text pairs.

👉 GPU support, which would enable accelerated inference and larger batch sizes.

👉 Deploy and implement our code targeting a distributed compute platform (Spark, Dask, Ray, Sagemaker etc.). This allows us to spread the workload across multiple compute instances.

#### Efficient Data Handling

100 million image-text pairs is a lot of data. To ensure data throughput and reducing the likely-hood of I/O bottlenecks we need to ensure we efficiently handle our data.

👉 Ensure the data we're processing is stored on a system that's highly available, scalable and performant. E.g. S3.

👉 Preprocess the data E.g. reduce large image sizes

👉 Bring compute close to the data. Rather than bringing lots of data to the compute, we could bring the compute to the data. Companies like Snowflake are starting to better support these types of database model integrations.


#### Model Optimisation

A lot of the time and memory footprint is directly correlated to our model.

👉 Consider model selection, would a smaller model that's slightly less accurate still suit our requirements?

👉 Can we reduce the model precision?

👉 Have we compared other similar models that are more performant?

## Challenge 3

> Briefly discuss how your method can be used to effectively curate data for text-to-image model training and provide an explanation

Curating lots of high-quality data is a difficult task, but this is also at the core of some of the most high-quality machine learning models we see today.

To curate a data for a text-to-image model we can leverage our similarity measurement. By having a quantifiable measure of how closely related an image is to it's associate description we are able to systematically curate large amounts of high-quality, diverse, ethical and accurate data.

👉 We can filter out image-text pairs that aren't similar, indicating poor data labelling quality.

👉 We can identify imbalances in the dataset, analyzing the distribution of scores across different categories.

👉 We can identify biases and enhance diversity in our dataset.

👉 We can identify errors in our dataset, e.g. negative similarity scores might indicate items are mislabelled.

# Experiment Planning Tool

We are planning the development of ROSE, a reflectometry experiment optimization tool.

The main approach of this tool will be based on the planning tool in this repo: $USER/git/analyzer/tree/main/analyzer_tools/planner

See the implementation phases are the end. When creating the web app, please consider how the app could serve as an extension/plug-in for the AuRE web app found here: $USER/git/aure

Here are some use-cases to address:

## Use-case 1: Determining best experimental conditions
This is the use-case found in the github repo above. From a give model, we want to choose optimize the subset of parameters that we can control (initial film thickness, etc...).
We aim at finding the values that maximize information gain. An added consideration here is that we may want to optimize information gain for a particular model parameter or set of parameters.
So we may be interested in the marginalized distribution.

For instance: Given a model for layer A on top of layer B on top of a silicon substrate, find the best thickness of layer B to be sensitive to changes in the thickness of layer A.

## Use-case 2: Plan experiment from textual description
From an input text (query.yaml) describing the sample geometry and the hypothesis to test in words, create a base refl1d model suggestion for the user, then in a second step run use-case one after letting the user select which parameters to optimize.


## Implementation Phases

- Phase 0: Implement package structure
- Phase 1: CLI for Use-case 1
- Phase 2: CLI for Use-case 2
- Phase 3: Flask web app to visualize the output of use-cases 1 and 2
- Phase 4: Extend flask app to interactively enter the information to start use-cases 1 and 2. Consider how this can be written as a plug-in to supplement $USER/git/aure, but make it standalone for this phase.


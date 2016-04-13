# browsing-behavior-reconstuction-paper

Reconstruction and analysis code for paper: Reconstructing Detailed Browsing Activities from Browser History

[Paper: Full](https://www.gkovacs.com/browsing-behavior-reconstuction-paper/reconstruct.pdf)

[Reconstruction and analysis code](https://github.com/gkovacs/browsing-behavior-reconstuction-analysis)

[LaTeX sources for paper](https://github.com/gkovacs/browsing-behavior-reconstuction-paper)

## Abstract

Users’ detailed browsing activity – such as what sites they are spending time on and for how long, and what tabs they have open and which one is focused at any given time – is useful for a number of research and practical applications. Gathering such data, however, requires that users install and use a monitoring tool over long periods of time. In contrast, browser extensions can gain instantaneous access months of browser history data. However, the browser history is incomplete: it records only navigation events, missing important information such as time spent or tab focused. In this work, we aim to reconstruct time spent on sites with only users’ browsing histories. We gathered three months of browsing history and two weeks of ground-truth detailed browsing activity from 185 participants. We developed a machine learning algorithm that predicts whether the browser window is focused and active at one second-level granularity with an F1-score of 0.84. During periods when the browser is active, the algorithm can predict which the domain the user was looking at with 76.2% accuracy. We can use these results to reconstruct the total time spent online for each user with an R2 value of 0.96, and the total time each user spent on each domain with an R2 value of 0.92.

## License

MIT

## Author

[Geza Kovacs](https://www.gkovacs.com)

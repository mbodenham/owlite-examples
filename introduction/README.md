# Introduction

Here we will introduce the basics of OwLite and it's capabilities.

Complete code can be found in `introduction.py`.
Let's use ResNet18 as our starting model.

1. Firstly code is set up using standard pytorch code. Here let's load the ResNet18 model using the torchvision library.

        import torch
        from torchvision.models import resnet18

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = resnet18(weights="DEFAULT")
        model.to(device)

        dummy_input = torch.randn(batch_size, 3, 224, 224)

2. Now let's add the required OwLite functions.

        import owlite

        owl = owlite.init(project="Introduction", baseline="resnet18")

        # Wrap model with owlite.convert
        model = owl.convert(model, dummy_input)

        owl.export(model) # Export model to ONNX
        owl.benchmark() # Benchmark model

3. The code can now be run inside the provided docker container.

        ./start_owlite
        cd introduction/ && python introduction.py

4. After completion benchmarking information will be displayed.

       OwLite [INFO] Baseline: resnet18
              Latency: 2.94397 (ms) on NVIDIA RTX A6000
              For more details, visit https://owlite.ai/project/detail/672bf5dc0bebf7120a583076

5. Following the link provided by OwLite allows for the results of the resnet18 model to be viewed.

![OwLite dashboard](./images/owlite_dash_1.webp "OwLite Dashboard")

6. Clicking on the orange + in the resnet18 row will allow us to begin a quantization experiment. Input a name for the experiment and click Enter.

![Creating an experiment](./images/owlite_dash_create_exp.webp "Creating an experiment")

7. The next page shows an overview of the models architecture.

![Model overview](./images/owlite_model_view.webp "Model overview")

8. Let's quantize the model using the recommended setting. Go Tools -> Recommended setting.

![Model tools](./images/owlite_model_tools.webp "Model tools")

9. Confirm the choice by clicking TRUST ME!

![Model recommended](./images/owlite_model_recommended.webp "Model recommended")

10. The recommended quantization setting have been loaded. Here the quantization setting of each layer can be adjusted, but lets leave everything at the recommended settings for this introduction. Click the orange save icon at the top of the model to save this setup.

![Model quantized](./images/owlite_model_quantized.webp "Model quantized")

11. Here the quantization has been saved and can now be tested by updating `introduction.py`.

![Model experiment](./images/owlite_model_experiment.webp "Model experiment")

12. Let's add the experiment to `introduction.py` by updating the `owlite.init` function.

        owl = owlite.init(project="Introduction", baseline="resnet18", experiment="first_quantization")

13. Run the experiment.

        python introduction.py

14. After the code has ran the results of the quantization experiment can be seen on the owlite dashboard.

![Owlite dashboard results](./images/owlite_dashboard_results.webp "Owlite dashboard results")

15. The recommended setting produce a **73% reduction in memory** and **2.3x speed up in inference latency**.



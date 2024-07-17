from manim import *
from manim_weighted_line import *

class MeasuringConnectivity(Scene):

    def construct(self):
        title = Title("Connectivity:")
        two_nodes_graph = Graph(vertices=["a" , "b"] , edges=[("a" , "b")] , labels=True , layout="partite"  , partitions=[["a"] , ["b"]] , vertex_config={
            "a" : {"fill_color" : RED_B , "radius" : 0.4},
            "b" : {"fill_color" : BLUE_B , "radius" : 0.4}
        })
        one_node_graph = Graph(["ab"]  , edges=[] , labels=True , vertex_config={"ab" : {"radius" : 0.5 , "fill_color" : color_gradient([RED_B , BLUE_B] , length_of_output=2)}})

        mean_measure = MathTex("""
                               (ab\\textsubscript{$x$} , ab\\textsubscript{$y$}) = 
                                    (
                                        \\frac{a\\textsubscript{$x$} + b\\textsubscript{$x$}}{2},
                                        \\frac{a\\textsubscript{$y$} + b\\textsubscript{$y$}}{2}
                                    )
                        """).next_to(one_node_graph , direction=DOWN)


        self.play(Write(title) , lag_ratio=0.001)
        self.play(Write(two_nodes_graph) , lag_ratio=0.01)
        self.wait()

        self.play(Transform(two_nodes_graph , one_node_graph) , lag_ratio=0.02)
        self.add(one_node_graph)
        self.remove(two_nodes_graph)
        self.wait()

        self.play(Write(mean_measure) , lag_ratio=0.05)
        self.wait()

        self.play(FadeOut(one_node_graph) , lag_ratio=0.05)
        self.play(Unwrite(mean_measure) , lag_ratio=0.02)
        self.play(Unwrite(title))
        self.wait()


class DescribingProblem(Scene):
    def construct(self):
        title = Title("Describing Problem:")
        one_node_graph = Graph(["ab"]  , edges=[] , labels=True , vertex_config={"ab" : {"radius" : 0.5 , "fill_color" : color_gradient([RED_B , BLUE_B] , length_of_output=2)}})
        
        x_mean_var = Variable(0.0 , MathTex("x") , num_decimal_places=2).next_to(one_node_graph , DOWN)
        y_mean_var = Variable(0.0 , MathTex("y") , num_decimal_places=2).next_to(x_mean_var, DOWN)

        conn_var = Variable(0.0 , MathTex("connectivity") , num_decimal_places=2).move_to(RIGHT * 4)
        conn_var.add_updater(lambda conn : conn.tracker.set_value(1/6 * x_mean_var.tracker.get_value() +  2/5 * y_mean_var.tracker.get_value()))

        problem_text = Text(
            "Problem: Find the shortest distance between two nodes" , 
            t2c={"Problem" : RED} , 
            t2g={"shortest distance" : (ORANGE , GREEN)} , 
            t2s={"shortest distance" : ITALIC},
            t2f={"Problem" : ULTRAHEAVY},
            font_size=25
        )

        regression_text = Text("Regression" , color=ORANGE, weight=HEAVY)
        distance_var = Variable(0.0 , MathTex("distance") , var_type=Integer , num_decimal_places=1).next_to(regression_text , direction=DOWN)

        classification_text = Text("Classification" , color=GREEN , weight=HEAVY)
        graph = Graph(
            vertices=[1 , 2 ,3 , 4 , 5 ,6] , 
            edges=[(1 , 2) , (2,  3) , (1 , 3) , (5 , 2) , (5, 4) , (4 , 6) , (4 , 3)] ,
            edge_type=WeightedLine,
            labels=True,
            layout_scale= 2.5,
            edge_config={
                (1 , 2) : {"weight" : 5},
                (2 , 3) : {"weight" : 2},
                (1 , 3) : {"weight" : 6},
                (5 , 2) : {"weight" : 1},
                (5 , 4) : {"weight" : 4},
                (4 , 6) : {"weight" : 2},
                (4 , 3) : {"weight" : 3},
            }
        ).shift(3 * RIGHT)

        regression_or_classification = Text("Regression Vs. Classification" , t2c={"Regression" : ORANGE , "Classification" : GREEN})
        solution_text = Text("Solution", gradient=color_gradient([GREEN , ORANGE] , length_of_output=2) , weight=BOLD)

        self.play(Write(title) , lag_ratio=0.004)
        self.play(Write(one_node_graph) , lag_ratio=0.004)
        self.wait()
        
        self.play(Write(x_mean_var) , Write(y_mean_var) , lag_ratio=0.02)
        self.play(Write(conn_var) , lag_ratio=0.002)
        self.wait()

        self.play(x_mean_var.tracker.animate.set_value(5) ,run_time=1 , rate_func=linear)
        self.play(y_mean_var.tracker.animate.set_value(5) ,run_time=1 , rate_func=linear)
        self.play(x_mean_var.tracker.animate.set_value(-5), run_time=1 , rate_func=linear)
        self.play(y_mean_var.tracker.animate.set_value(-5), run_time=1 , rate_func=linear)

        self.play(FadeOut(x_mean_var , shift=DOWN) , FadeOut(y_mean_var , shift=DOWN) , FadeOut(conn_var , shift=RIGHT) , lag_ratio=0.002)
        self.play(FadeOut(one_node_graph , lag_ratio=0.002))

        self.play(Write(problem_text))
        self.wait()

        self.play(Transform(problem_text , regression_text) , lag_ratio=0.005)
        self.add(regression_text)
        self.remove(problem_text)
        self.play(Create(distance_var) , lag_ratio=0.05)
        self.play(distance_var.tracker.animate.set_value(50) , run_time=3, rate_func=linear)
        self.wait()

        self.play(FadeOut(regression_text) , FadeOut(distance_var , shift=DOWN) , lag_ratio=0.002)

        self.play(Write(classification_text) , lag_ratio=0.002)
        self.play(classification_text.animate.move_to(3 * LEFT), Create(graph) , lag_ratio=0.002)
        self.wait()

        self.play(FadeOut(classification_text , shift=LEFT) , FadeOut(graph , shift=RIGHT))
        self.wait()
        
        self.play(Write(regression_or_classification) , lag_ratio=0.002)
        self.wait()
        self.play(Transform(regression_or_classification , solution_text) , lag_ratio=0.002)
        self.add(solution_text)
        self.remove(regression_or_classification)
        self.wait()

        self.play(FadeOut(solution_text))
        self.play(Unwrite(title))
        self.wait()

class LinearNN:
    def __init__(self , n_layers : list) -> None:
        self.n_layers = n_layers
        self.graph =self._build()

    def getVertices(self, layer : int) -> VGroup:
        vertices_cum = sum([n_vertices for n_vertices in self.n_layers[:layer - 1]])

        start = vertices_cum + 1
        end = start + self.n_layers[layer - 1]
    
        return VGroup(*[self.graph.vertices[i] for i in range(start , end)])
    
    def get_input_size(self):
        return self.n_layers[0]
    
    def get_output_size(self):
        return self.n_layers[len(self.n_layers) - 1]

    def getEdges(self, edges_layer : int) -> VGroup:
        vertices_cum = sum([n_vertices for n_vertices in self.n_layers[:edges_layer - 1]])

        start = vertices_cum + 1
        end = start + self.n_layers[edges_layer - 1]

        start_layer = [i for i in range(start , end)]
        end_layer = [i for i in range(end , end + self.n_layers[edges_layer])]

        return VGroup(*[self.graph.edges[(i, j)] for i in start_layer for j in end_layer])


    def _build(self) -> Graph:
        edges = []
        partitions = []
        c = 0
        vertices = np.arange(1 , sum(self.n_layers) + 1)
        
        for i in self.n_layers:
            partitions.append(list(range(c + 1 , c + 1 + i)))
            c += i

        for i, v in enumerate(self.n_layers[1:]):
            last = sum(self.n_layers[:i+1])
            for j in range(v):
                for k in range(last - self.n_layers[i], last):
                    edges.append((k + 1, j + last + 1))

        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=2,
            vertex_config={'radius': 0.20},
        )

        return graph
    
class ExplainNN:

    class LayerType: 
        INPUT_LAYER  = 0
        HIDDEN_LAYER = 1
        OUTPUT_LAYER = 2

    class ModelType:
        LINEAR_MODEL = MathTex("\\alpha X + \\beta")


    class ActivationFunctionType:
        SIGMOID = 0
        RELU = 1

    class Node:
        def __init__(self , content : Mobject , content_radius : int = 0.4 , circle_radius : int = 0.6 , node_color : str = RED , node_opacity : float = 1.0) -> None:
            self.content = content
            self.content.surround(Circle(radius=content_radius))
            self.node = Circle(radius=circle_radius , color=node_color , fill_opacity=node_opacity)
            self.node.add(self.content)

        def get_node(self) -> Circle:
            return self.node
        
        def get_contet(self) -> Mobject:
            return self.content
        
    def __init__(self , layers : list) -> None:
        self.layers = []
        self.nn = VGroup()
        self.nn_layers = []
        self.edges = []
        self.nn_edges = []

        for i_layer in range(len(layers)):
            layer = layers[i_layer]
            new_layer = None
            opacity = layer.get("opacity" , 1.0)
            color = layer.get("color" , RED)
            if layer["type"] == ExplainNN.LayerType.INPUT_LAYER:
                new_layer = [
                    ExplainNN.Node(
                        MathTex(
                            "x\\textsubscript{%d}"%(i+1)
                        ) ,  
                        node_opacity=opacity , 
                        node_color=color, 
                        content_radius=0.2 , 
                        circle_radius=0.4
                    ).get_node() 
                    for i in range(layer["size"])
                ]

            elif layer["type"] == ExplainNN.LayerType.HIDDEN_LAYER: 
                model = layer.get("model" , ExplainNN.ModelType.LINEAR_MODEL)
                new_layer = [
                    ExplainNN.Node(
                        model.copy() , 
                        node_opacity=opacity,
                        node_color=color
                    ).get_node() 
                    for _ in range(layer["size"])
                ]

            elif layer["type"] == ExplainNN.LayerType.OUTPUT_LAYER:
                mobj = layer.get("mobj" , MathTex("output"))
                new_layer = [
                    ExplainNN.Node(
                        mobj,
                        node_opacity=opacity,
                        node_color=color
                    ).get_node() for _ in range(layer["size"])
                ]

            nn_layer = VGroup(*new_layer).arrange(direction=DOWN , buff=0.3)

            if i_layer > 0:
                nn_layer.next_to(self.nn_layers[len(self.nn_layers) - 1] , RIGHT, buff=0.4)

            self.nn_layers.append(nn_layer)
            self.layers.append(new_layer)

        for i_layer in range(len(self.layers) - 1):
            layer = self.layers[i_layer]
            next_layer = self.layers[i_layer + 1]
            layer_edges = []

            for node in layer:        
                for next_node in next_layer:
                    layer_edges.append(Line(node , next_node))

            self.edges.append(layer_edges)
            self.nn_edges.append(VGroup(*layer_edges))
                
                
        self.nn= VGroup(VGroup(*self.nn_edges) , VGroup(*self.nn_layers)).center()

    def get_nn(self) -> VGroup:
        return self.nn

    def get_layer_vertices(self, layer) -> list:
        return self.layers[layer - 1]
    
    def get_vertices(self) -> list:
        return self.layers
    
    def get_edges(self) -> list:
        return self.edges
    
    def get_layer_edges(self, layer) -> list:
        return self.edges[layer - 1]

    def get_layer_as_vertices(self, layer) -> VGroup:
        return self.nn_layers[layer - 1]
    
    def get_layer_as_edges(self, layer) -> VGroup:
        return self.nn_edges[layer - 1]


class MathimaticalModel(Scene):
    
    def construct(self):
        title = Title("Mathimatical Model:")
        nn = LinearNN([5 , 3 , 3 , 1])
        hyper_parameters_text = Text("hyper-parameters" , color=YELLOW , slant=ITALIC , font_size=20).next_to(nn.graph , direction=DOWN)

        input_layer_rect = SurroundingRectangle(nn.getVertices(1))
        input_layer_size = MathTex("m").next_to(input_layer_rect, direction=LEFT)
        input_layer_text = Text("Input Layer" , font_size=20 , color=WHITE).next_to(input_layer_rect , direction=DOWN)
        input_matrix = MobjectMatrix(
            [
                MathTex("emb\\textsubscript{1}" , font_size=30) , 
                MathTex("emb\\textsubscript{2}" , font_size=30) , 
                MathTex("......" , font_size=30)  , 
                MathTex("emb\\textsubscript{m}" , font_size=30)
        ]).next_to(input_layer_rect , direction=LEFT)

        output_layer_rect=  SurroundingRectangle(nn.getVertices(4))
        output_layer_text = Text("distance" , slant=ITALIC , font_size=20).next_to(output_layer_rect , direction=RIGHT)

        explain_nn = ExplainNN([{
            "size" : 5 , 
            "type" : ExplainNN.LayerType.INPUT_LAYER
        },
        {
            "size" : 3,
            "type" : ExplainNN.LayerType.HIDDEN_LAYER,
            "color": DARK_BLUE,
        } ,
        {
            "size" : 3,
            "type" : ExplainNN.LayerType.HIDDEN_LAYER,
            "color": DARK_BLUE,
        } , 
        {
            "size" : 1,
            "output" : MathTex("distance"),
            "type" : ExplainNN.LayerType.OUTPUT_LAYER,
            "color": RED,
        }])

        sequence_layers_text = Text(
            "Sequence of Linear Models" , 
            slant=ITALIC , 
            weight=BOLD ,
            font_size=20,
            color=YELLOW , 
        ).next_to(explain_nn.get_nn() , DOWN)

        relu_function = MathTex(
            """\\mbox{ReLU} = 
            \\left\\{ 
            \\begin{array}{ccc} 
            x & \\mbox{for} & x >= 0 \\\ 
            0 & \\mbox{for} & x < 0  
            \\end{array} \\right.""").surround(Rectangle(height=4 , width=4)).next_to(explain_nn.get_nn() , direction=DOWN)
        
        axes = Axes(
            x_range=[-5 , 5 , 1] , 
            y_range=[0 , 7 , 1] , 
            axis_config={"include_numbers" : True}
        ).surround(Rectangle(height=5 , width=5)).shift(4 * RIGHT)

        relu_plot = axes.plot(lambda x : x if x >= 0 else 0 , x_range=[-3 , 3], use_smoothing=False).set_color(RED)

        rmsprop_axes = Axes(
            x_range = [-5 , 5 , 1],
            y_range = [0 , 25 , 5],
            axis_config={"include_numbers" : True}
        ).surround(Rectangle(height=5, width=5)).shift(3 * LEFT)
        sgd_axes = rmsprop_axes.copy().surround(Rectangle(height=5, width=5)).shift(3 * RIGHT)

        loss_fn = lambda x : x**2
        rmsLoss_plot = rmsprop_axes.plot(loss_fn , x_range=[-5 , 5] , use_smoothing=True).set_color(WHITE)
        sgdLoss_plot = sgd_axes.plot(loss_fn , x_range=[-5 , 5] , use_smoothing=True).set_color(WHITE)


        learning_algo_text = Text("Learning Algorithm")
        rmsProp_text = Text("RMSProp" , color=ORANGE, font_size=30).next_to(learning_algo_text , direction=DOWN)


        sgd_x_input = [-5 , 4.5 , -3.8 , 3.1 , -2.5 , 1.8 , -1.7 , 1.6 , -1.55 , 1.34 , 1.30, 1.28, 1.23]
        rms_x_input = [-5 , 4.8 , -4   , 3.3 , -2.7 , 1.9 , -1.78, 1.5 , -1.5  , 1.10 , 0.9 , 0.6 , 0.2]
        
        sgd_points = [Dot(sgd_axes.coords_to_point(sgd_x , sgd_x ** 2) , color=GOLD_A , radius=0.05) for sgd_x in sgd_x_input]
        rms_points = [Dot(rmsprop_axes.coords_to_point(rms_x , rms_x ** 2), color=PURPLE_B , radius=0.05) for rms_x in rms_x_input]

        sgd_lines = [Line(start=sgd_points[i] , end=sgd_points[i + 1] , color=GOLD_A) for i in range(len(sgd_points) - 1)]
        rms_lines = [Line(start=rms_points[i] , end=rms_points[i + 1] , color=PURPLE_B) for i in range(len(rms_points) - 1)]

        optim_method_text = Text("Optimization With MSE" , color=BLUE_B , font_size=20).shift(3 * DOWN)
        adaptive_lr_text = Text("Adaptive Learning Rate" , font_size=20).next_to(rmsprop_axes , direction=DOWN)
        sgd_lr_text = Text("Fixed Learning Rate" , font_size=20).next_to(sgd_axes , direction=DOWN)

        
        loss_fn_text = Text("Loss Function" , color=RED_B)
        poisson_loss = MathTex("Loss = -\\log p(y|X,w) = \\sum_{t} \\mu \\textsuperscript{$t$} + y \\textsuperscript{$t$} \\log \\mu \\textsuperscript{$t$}" , font_size=40).next_to(loss_fn_text , direction=DOWN)
        crossEntropy_loss = MathTex("H(p,q) = -\\sum_{x \\in distances} p(x) \\log q(x)" , font_size=40).next_to(loss_fn_text , direction=DOWN)

        other_params_text = Text("Other Hyper-Parameters")
        batch_size_text = Text("Bactch Size",  font_size=20 , color=GOLD_A).next_to(other_params_text , direction=UR).shift(2 * LEFT)
        dropout_text = Text("Droup Out",  font_size=20 , color=GREEN_B).next_to(other_params_text , direction=UL).shift(2 * RIGHT)
        min_max_lr_text = Text("Min. & Max LR",  font_size=20 , color=BLUE_B).next_to(other_params_text , direction=DL).shift(2 * RIGHT)
        patience_text = Text("Early Stopping\nPatience", should_center=True , font_size=20 , color=PURPLE_B).next_to(other_params_text , direction=DR).shift(2 * LEFT)


        self.play(Write(title) , lag_ratio=0.002)
        self.play(Create(nn.graph) , lag_ratio=0.002)
        self.wait()

        self.play(Write(hyper_parameters_text))
        self.wait()
        self.play(FadeOut(hyper_parameters_text , shift=DOWN))

        self.play(Write(input_layer_rect) , Write(input_layer_text) , lag_ratio=0.002)
        self.wait()
        self.play(Write(input_matrix) , lag_ratio=0.002)
        self.wait()
        self.play(Transform(input_matrix , input_layer_size) , lag_ratio=0.002)
        self.add(input_layer_size)
        self.remove(input_matrix)
        self.wait()

        self.play(Unwrite(input_layer_size) , Unwrite(input_layer_text) ,  Uncreate(input_layer_rect) , lag_ratio=0.002)

        self.play(Write(output_layer_rect) , lag_ratio=0.002)
        self.wait()
        self.play(Write(output_layer_text) , lag_ratio=0.002)
        self.wait()
        self.play(Unwrite(output_layer_text) , Uncreate(output_layer_rect) , lag_ratio=0.002)
        self.wait()

        self.play(FadeOut(nn.graph) , lag_ratio=0.002)


        self.play(Write(explain_nn.get_nn()) , lag_ratio=0.002)
        self.wait()

        self.play(
            *[vertex.animate.set_stroke_color(BLUE_B) for vertex in explain_nn.get_layer_vertices(2)],
            *[vertex.animate.set_stroke_color(BLUE_B) for vertex in explain_nn.get_layer_vertices(3)] , 
            lag_ratio=0.002
        )

        self.play(Write(sequence_layers_text) , lag_ratio=0.02)
        self.wait()
        self.play(FadeOut(sequence_layers_text) , lag_ratio=0.02)
        self.wait()

        self.play(
            *[edge.animate.set_color(GREEN) for edge in explain_nn.get_layer_edges(2)] ,
            *[edge.animate.set_color(GREEN) for edge in explain_nn.get_layer_edges(3)] ,
            lag_ratio=0.002)
        self.wait()
        self.play(Write(relu_function) , lag_ratio=0.02)
        self.wait()

        self.play(explain_nn.get_nn().animate.shift(3 * LEFT) , relu_function.animate.shift(3 * LEFT) , lag_ratio=0.002)
        self.play(Create(axes), Write(relu_plot) , lag_ratio=0.002)
        self.wait()

        self.play(FadeOut(axes , relu_plot , shift=RIGHT) , FadeOut(explain_nn.get_nn(), shift=LEFT) , FadeOut(relu_function , shift=DOWN))
        self.wait()

        self.play(Write(learning_algo_text) , lag_ratio=0.002)
        self.wait()
        self.play(Write(rmsProp_text), lag_ratio=0.002)
        self.wait()

        self.play(FadeOut(learning_algo_text , rmsProp_text) , lag_ratio=0.002)
        self.wait()

        self.play(Write(optim_method_text) , lag_ratio=0.002)
        self.play(Create(rmsprop_axes) , Create(sgd_axes) , lag_ratio=0.002)
        self.play(Write(adaptive_lr_text) , Write(sgd_lr_text) , lag_ratio=0.002)
        self.play(Write(rmsLoss_plot) , Write(sgdLoss_plot) , lag_ratio=0.002)

        self.play(
            *[Create(point) for point in rms_points],
            *[Create(point) for point in sgd_points],
            lag_ratio=0.002
        )

        self.wait()

        for i in range(len(rms_lines)):
            self.play(
                Create(rms_lines[i] , lag_ratio=0.002),
                Create(sgd_lines[i] , lag_ratio=0.002)
            )

        self.wait()

        self.play(FadeOut(sgd_lr_text) , FadeOut(sgdLoss_plot) , FadeOut(sgd_axes) , FadeOut(*sgd_lines) , FadeOut(*sgd_points))
        self.play(FadeOut(adaptive_lr_text) , FadeOut(rmsLoss_plot) , FadeOut(rmsprop_axes) , FadeOut(*rms_lines) , FadeOut(*rms_points))
        self.play(FadeOut(optim_method_text))
        self.wait()

        self.play(Write(loss_fn_text),lag_ratio=0.02)
        self.wait()
        
        self.play(Write(poisson_loss) , lag_ratio=0.02)
        self.wait()

        self.play(Transform(poisson_loss , crossEntropy_loss))
        self.add(crossEntropy_loss)
        self.remove(poisson_loss)
        self.wait()
        self.play(FadeOut(crossEntropy_loss) , FadeOut(loss_fn_text) , lag_ratio=0.002)
        self.wait()

        self.play(Write(other_params_text) , lag_ratio=0.002)
        self.wait()
        self.play(Write(batch_size_text) , lag_ratio=0.002)
        self.wait()
        self.play(Write(dropout_text) , lag_ratio=0.002)
        self.wait()
        self.play(Write(min_max_lr_text) , lag_ratio=0.002)
        self.wait()
        self.play(Write(patience_text) , lag_ratio=0.002)
        self.wait()

        self.play(Unwrite(other_params_text) , Unwrite(batch_size_text) , Unwrite(dropout_text) , Unwrite(min_max_lr_text) , Unwrite(patience_text))
        self.play(Unwrite(title))
        self.wait()



---------------------------------------------
Empregando CMake para construir proxectos C++
---------------------------------------------

Empregar CMake para compilar código é unha boa elección. Aportamos esta breve guía para aquela xente que
nunca empregou esta ferramenta. Parece un pouco difícil, pero unha vez collida a mecánica, o proceso global de onfiguración, compilado, linkado e execución faise extraordinariamente doado. Para máis información: https://cmake.org/cmake/help/latest/guide/tutorial/index.html

PD. - Nós aportamos os ficheiros cmake para que non teñas que escribilos ti mesmo senon empregalos e modificalos ao teu antoxo. Non obstante, se tes interese consulta a documentación de CMake.

Step 1 : Creando o ficheiro CMakeLists.txt

Para cada proxecto, aprenderemos a crear un ficheiro CMake.
CMake dalle a configuración do teu proxecto (rutas aos ficheiros include e librarias) a partir dun ficheiro chamado CMakeLists.txt. Este debe estar situado no mesmo directorio onde tes o ficheiros fonte C++

Abaixo, amoso o aspecto do ficheiro CMakeLists.txt file. Cada liña detes ficheiro ten unha explicación 
que se escribe enriba de cada renglón: 

############# Exemplo do ficheiro CMakeLists.txt #############

# Definimos a mínima version de CMake que pode executar este ficheiro
cmake_minimum_required(VERSION 2.8.12)

# Asignamos un nome ao noso proxecto
PROJECT(meuProxecto)

## CMake ten que coñecer onde esta instalada a libreria OpenCV para poder compilar e linkar o noso ficehiro. Non obstante,
# CMake busca os ficheiros OpenCVConfig.cmake e OpenCVConfig-version.cmake para configurar a OpenCV.
# Temos DOUS xeitos de indicarlle a CMake como atopar estes ficheiros de OpenCV:
	# 1. Definir directamente unha variable de entorno chamada OpenCV_DIR e darlle a ruta ao cartafol onde se atopan os 
	# ficheiros OpenCVConfig.cmake and OpenCVConfig-version.cmake
	# 2. Asignar estas variables de OpenCV_DIR no ficheiro CMakeLists.txt.

# Nós seguiremos a segunda opción. Deste xeito podes ter multiples versións de OpenCV
# instaladas na túa máquina e establecer a que queres empregar en cada proxecto.

# Indicasmoslle que atope no meu ordenador o paquete OpenCV
find_package( OpenCV REQUIRED )

# Asignamos a ruta a variable indicada abaixo para os Includes de OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS})

# Supoñamos que temos un ficheiro cpp chamado meuCodigo.cpp
# Imos a escribir a regra de compilación e linkado.
# En ADD_EXECUTABLE o primeiro nome é o executable que se xerará cando compilemos e
# linkemos o ficheiro fonte chamdo meuCodigo.cpp 
ADD_EXECUTABLE(meuCodigo.out meuCodgo.cpp)

# Isto é todo o que precisamos para compilar o código C++. Non obstante, se temos varios ficheiros
# C++ para compilar e linkar podemos escribir unha macro de tal xeito que non nos vexamos obrigados a
# escribir esta liña tantas veces como ficheiros teña o meu proxecto. 
# Definimos unha Macro nomeada add_ficheiro e engadimos as regras de compilación.
MACRO(add_ficheiro nome)
  ADD_EXECUTABLE(${nome} ${nome}.cpp)
  TARGET_LINK_LIBRARIES(${nome} ${OpenCV_LIBS} )
ENDMACRO()

# Agora podemos empregar esta macro para compilar os ficheiros
add_ficheiro(meuCodigo2)
add_ficheiro(meuCodigo3)


Paso 2: Construimos o proxecto con CMake
Compilamos o proxecto empregano a lista de CMakeLists.txt que debemos manter no mesmo cartafol onde estan os ficheoros cpp.
Agora abre a terminal e executa os seguintes comandos dende consola para compilar o proxecto.

mkdir build
cd build


Configurando o proxecto empregando CMake (Ubuntu)
Decimoslle a CMake que configure e xere os ficheiros do proxecto. 
.. denota que CMake debe buscar CMakeLists.txt no directorio padre do directorio build.
Para Ubuntu cmake detecta e emprega a toolchain de C++ instalada no teu sistema (probablemente sexa a de gcc).

cmake ../src

Construimos o proxecto empregando CMake
Decimoslle a CMake que consgtrúa o proxecto en modo Release. Tamén podemos facelo en modo Debug para configuración.

cmake --build . --config Release

Cando a construccion está completa, os executables xeraranse no mesmo cartafol build.

Eecutamos o programa compilado e linkado:
Podemos ir ao directorio do proxecto (padre de build) para executar o programa. É importarte que indiques a ruta relativa:
./build/meu_programa_1
./build/meu_progama_2


OLLO: Asegurate de que o ficheiro xerado *.out ten permisos de execución (chmod 777 nome_fich.out)




----------------------------------------------------------------------------------------------
----------------------------Outra opcion menos elegante --------------------------------------
----------------------------------------------------------------------------------------------
Outra opción é facer unha macro para a shell chamando directamente ao compilador gcc. Se compilamos OpenCV directamente dende o código fonte, é activamos a opción PKG_CONFIG, logo podemos invocalo diretamente dende a liña do compilador. Esta opción introduce automaticamente todas os includes e librarias de OpenCV necesarias para compilar e linkar o meu código non OpenCV. Por exemplo, se o meu programa se chamase meu_Codigo.cpp, a instrucción que precisaria sería:

g++ -std=c++11 meu_Codigo.cpp `pkg-config --libs --cflags opencv` -o meu_Codigo.out









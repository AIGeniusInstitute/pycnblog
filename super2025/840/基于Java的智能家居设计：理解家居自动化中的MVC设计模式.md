                 

# 基于Java的智能家居设计：理解家居自动化中的MVC设计模式

> 关键词：智能家居,Java,家居自动化,MVC,设计模式,面向对象编程

## 1. 背景介绍

### 1.1 问题由来
随着物联网技术的迅猛发展，智能家居系统已经成为现代家庭生活中不可或缺的一部分。智能家居系统通过将家庭中的各种设备和系统互联，实现对家居环境的智能化管理和控制。然而，如何设计一个高效、易扩展的智能家居系统，实现复杂功能的同时保持系统的简洁性和可维护性，是当前智能家居开发中面临的重大挑战。

### 1.2 问题核心关键点
为了解决这一问题，本文将重点探讨MVC（Model-View-Controller）设计模式在家居自动化系统中的应用。MVC是一种经典的软件设计模式，用于分离内层数据模型和业务逻辑与外层用户界面和交互逻辑，使其相互独立，易于维护和扩展。通过MVC设计模式，智能家居系统可以更加清晰地划分功能模块，提高开发效率，降低维护成本。

### 1.3 问题研究意义
深入理解MVC设计模式在家居自动化系统中的应用，对构建高效、易扩展、易维护的智能家居系统具有重要意义。MVC模式能够帮助开发者更清晰地进行系统设计和功能划分，提高系统的可重用性和可维护性，使智能家居系统能够快速适应不断变化的需求，支持系统的长期发展和迭代升级。

## 2. 核心概念与联系

### 2.1 核心概念概述

- **智能家居系统**：通过互联网将家庭中的各种设备和系统互联，实现对家居环境的智能化管理和控制。包括温度控制、照明调节、安全监控、智能门锁等众多功能。

- **MVC设计模式**：一种经典的软件设计模式，用于分离内层数据模型和业务逻辑与外层用户界面和交互逻辑，使其相互独立，易于维护和扩展。MVC将系统的功能分为三个部分：Model（数据模型）、View（用户界面）和Controller（控制器）。

- **面向对象编程（OOP）**：一种编程范式，将数据和行为组织成对象，通过对象之间的交互来实现系统的功能。OOP有助于构建模块化、可扩展和可维护的软件系统。

- **设计模式**：一种软件设计原则和经验总结，用于解决特定场景下的软件设计问题。设计模式可以提高代码的复用性、可维护性和可扩展性。

- **家居自动化技术**：通过传感器、控制器和通信技术，实现对家居设备的自动控制和智能化管理。包括自动化照明、温控、安防、家庭娱乐等众多功能。

这些核心概念构成了智能家居系统设计的基础，而MVC设计模式则是实现系统模块化和功能划分的关键手段。以下Mermaid流程图展示了这些核心概念之间的联系：

```mermaid
graph LR
    A[智能家居系统] --> B[家居自动化技术]
    A --> C[MVC设计模式]
    A --> D[面向对象编程]
    B --> E[传感器]
    B --> F[控制器]
    B --> G[通信技术]
    C --> H[数据模型]
    C --> I[用户界面]
    C --> J[控制器]
    D --> K[对象]
    D --> L[继承]
    D --> M[封装]
    E --> N[环境监测]
    F --> O[自动控制]
    G --> P[数据传输]
    H --> Q[家居数据]
    I --> R[用户交互]
    J --> S[业务逻辑]
    K --> T[模块]
    L --> U[复用]
    M --> V[隔离]
    N --> W[采集数据]
    O --> X[执行命令]
    P --> Y[信息传递]
    Q --> Z[系统数据]
    R --> $[交互展示]
    S --> &[系统逻辑]
    T --> [][组件]
    U --> [][重用]
    V --> [][隔离风险]
    W --> [][数据采集]
    X --> [][控制指令]
    Y --> [][信息处理]
    Z --> [][系统数据]
    $ --> [][用户展示]
```

### 2.2 概念间的关系

这些核心概念之间存在着紧密的联系，共同构成了智能家居系统的设计框架。以下Mermaid流程图展示了这些概念之间的关系：

```mermaid
graph LR
    A[智能家居系统] --> B[家居自动化技术]
    A --> C[MVC设计模式]
    A --> D[面向对象编程]
    B --> E[传感器]
    B --> F[控制器]
    B --> G[通信技术]
    C --> H[数据模型]
    C --> I[用户界面]
    C --> J[控制器]
    D --> K[对象]
    D --> L[继承]
    D --> M[封装]
    E --> N[环境监测]
    F --> O[自动控制]
    G --> P[数据传输]
    H --> Q[家居数据]
    I --> R[用户交互]
    J --> S[业务逻辑]
    K --> T[模块]
    L --> U[复用]
    M --> V[隔离]
    N --> W[采集数据]
    O --> X[执行命令]
    P --> Y[信息传递]
    Q --> Z[系统数据]
    R --> $[交互展示]
    S --> &[系统逻辑]
    T --> [][组件]
    U --> [][重用]
    V --> [][隔离风险]
    W --> [][数据采集]
    X --> [][控制指令]
    Y --> [][信息处理]
    Z --> [][系统数据]
    $ --> [][用户展示]
```

### 2.3 核心概念的整体架构

最后，我们用一个综合的流程图来展示这些核心概念在家居自动化系统中的整体架构：

```mermaid
graph TB
    A[传感器] --> B[数据模型]
    A --> C[控制器]
    A --> D[通信技术]
    B --> E[家居数据]
    C --> F[自动控制]
    D --> G[数据传输]
    E --> H[系统数据]
    F --> I[执行命令]
    G --> J[信息传递]
    H --> K[数据处理]
    I --> L[控制逻辑]
    J --> M[信息处理]
    K --> N[分析结果]
    L --> O[业务逻辑]
    M --> P[数据输出]
    N --> Q[决策支持]
    O --> R[系统功能]
    P --> S[用户界面]
    Q --> T[功能模块]
    R --> U[用户交互]
    S --> V[展示界面]
    T --> W[组件]
    U --> X[用户操作]
    V --> Y[交互展示]
    W --> Z[模块组件]
    X --> [][用户命令]
    Y --> [][交互响应]
    Z --> $[模块功能]
    $ --> [][用户反馈]
```

这个综合流程图展示了从传感器采集数据到最终用户反馈的完整过程，通过MVC设计模式将系统功能模块化和逻辑分离，提高了系统的可维护性和可扩展性。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

基于MVC设计模式的智能家居系统，其核心算法原理可以简述如下：

- **数据模型（Model）**：负责处理家居设备的感知数据，包括温度、湿度、烟雾、门窗状态等环境信息。数据模型将采集到的数据进行处理、存储和分析，提供系统决策所需的原始数据。

- **用户界面（View）**：负责与用户进行交互，展示家居系统的状态和控制界面。用户界面可以包括智能手机APP、Web页面、语音助手等。

- **控制器（Controller）**：负责处理用户输入的控制指令，并调用数据模型进行处理。控制器是系统的核心逻辑单元，负责协调数据模型和用户界面之间的数据交换和逻辑控制。

### 3.2 算法步骤详解

以下是智能家居系统基于MVC设计模式的具体操作步骤：

**Step 1: 定义数据模型**

定义数据模型是智能家居系统的基础。数据模型应该能够处理家庭环境中的各种数据，包括温度、湿度、烟雾、门窗状态等。可以使用Java中的数据结构和类来定义数据模型。例如，可以定义一个`EnvironmentData`类，用于存储和处理家庭环境数据：

```java
public class EnvironmentData {
    private double temperature;
    private double humidity;
    private boolean smokeDetected;
    private boolean doorOpen;
    // getters and setters
}
```

**Step 2: 设计用户界面**

设计用户界面是智能家居系统的关键。用户界面应该易于使用，能够直观展示家居系统的状态和控制选项。可以使用Java Swing或JavaFX等GUI工具包来设计用户界面。例如，可以设计一个简单的GUI界面，用于展示家庭环境数据和控制家居设备：

```java
import javax.swing.*;
import java.awt.*;

public class HomeInterface extends JFrame {
    private JLabel temperatureLabel;
    private JLabel humidityLabel;
    private JButton controlButton;
    
    public HomeInterface() {
        // 创建界面元素并设置布局
        temperatureLabel = new JLabel("Temperature: ");
        humidityLabel = new JLabel("Humidity: ");
        controlButton = new JButton("Control Devices");
        
        // 设置布局和显示界面
        setLayout(new GridLayout(3, 2));
        add(temperatureLabel);
        add(new JLabel(String.format("%.1f", temperature)));
        add(humidityLabel);
        add(new JLabel(String.format("%.1f", humidity)));
        add(controlButton);
        
        // 添加控制按钮事件处理
        controlButton.addActionListener(e -> {
            // 处理控制指令
            // ...
        });
    }
    
    public static void main(String[] args) {
        new HomeInterface().setVisible(true);
    }
}
```

**Step 3: 实现控制器**

实现控制器是智能家居系统的核心。控制器负责接收用户输入的控制指令，并调用数据模型进行处理。控制器应该具有良好的可扩展性和可维护性，以便添加新的家居设备或功能。可以使用Java中的面向对象编程和设计模式来实现控制器。例如，可以定义一个`HomeController`类，用于处理用户控制指令：

```java
public class HomeController {
    private EnvironmentData environmentData;
    
    public HomeController(EnvironmentData environmentData) {
        this.environmentData = environmentData;
    }
    
    public void controlDevices(String command) {
        if (command.equals("heat")) {
            environmentData.setTemperature(environmentData.getTemperature() + 1);
        } else if (command.equals("cool")) {
            environmentData.setTemperature(environmentData.getTemperature() - 1);
        } else if (command.equals("vent")) {
            environmentData.setDoorOpen(!environmentData.isDoorOpen());
        }
    }
}
```

**Step 4: 连接数据模型、用户界面和控制器**

连接数据模型、用户界面和控制器是智能家居系统的最后一步。使用MVC设计模式，数据模型、用户界面和控制器之间的数据交换和逻辑控制应该互相独立。可以使用Java中的事件监听器和数据绑定技术来实现这一步骤。例如，可以将用户界面与数据模型和控制器连接起来：

```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

public class HomeApplication extends JFrame {
    private EnvironmentData environmentData;
    private HomeController homeController;
    
    public HomeApplication() {
        // 创建数据模型和控制器
        environmentData = new EnvironmentData();
        homeController = new HomeController(environmentData);
        
        // 创建界面元素并设置布局
        JLabel temperatureLabel = new JLabel("Temperature: ");
        JLabel humidityLabel = new JLabel("Humidity: ");
        JButton controlButton = new JButton("Control Devices");
        
        // 设置布局和显示界面
        setLayout(new GridLayout(3, 2));
        add(temperatureLabel);
        add(new JLabel(String.format("%.1f", environmentData.getTemperature())));
        add(humidityLabel);
        add(new JLabel(String.format("%.1f", environmentData.getHumidity())));
        add(controlButton);
        
        // 添加控制按钮事件处理
        controlButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                homeController.controlDevices("heat");
            }
        });
        
        // 监听环境数据变化事件
        environmentData.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                if ("temperature".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                } else if ("humidity".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                } else if ("doorOpen".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                }
            }
        });
    }
    
    public static void main(String[] args) {
        new HomeApplication().setVisible(true);
    }
}
```

### 3.3 算法优缺点

基于MVC设计模式的智能家居系统具有以下优点：

- **模块化和可扩展性**：MVC设计模式将系统功能模块化和逻辑分离，便于添加新的家居设备或功能，提高系统的可扩展性。

- **易于维护和调试**：MVC设计模式使数据模型、用户界面和控制器之间的交互透明化，便于维护和调试。

- **良好的用户体验**：用户界面设计简洁直观，易于使用，能够直观展示家居系统的状态和控制选项。

- **高效的数据处理**：数据模型负责处理家居设备的感知数据，能够高效存储和分析数据，提供系统决策所需的原始数据。

然而，基于MVC设计模式的智能家居系统也存在以下缺点：

- **学习曲线较陡峭**：MVC设计模式涉及面向对象编程和设计模式，需要一定的编程经验和设计思维。

- **开发成本较高**：MVC设计模式虽然提高了系统的可扩展性和可维护性，但也增加了系统的开发和维护成本。

- **性能瓶颈**：MVC设计模式中，数据模型和控制器之间的数据交换和逻辑控制需要通过事件监听器和数据绑定实现，可能存在性能瓶颈。

- **界面耦合度较高**：用户界面和数据模型之间的耦合度较高，可能导致界面变化影响系统功能。

### 3.4 算法应用领域

基于MVC设计模式的智能家居系统可以应用于各种家居自动化场景，包括：

- **智能照明**：通过传感器和控制器，实现对家中灯光的智能控制，根据时间、天气、用户行为等因素自动调整灯光亮度和颜色。

- **智能温控**：通过传感器和控制器，实现对家中空调、暖气等设备的智能控制，根据温度、湿度、用户偏好等因素自动调整温度和湿度。

- **智能安防**：通过传感器和控制器，实现对家中门窗、摄像头等设备的智能控制，根据异常情况自动报警或采取措施。

- **智能娱乐**：通过传感器和控制器，实现对家中音响、电视等设备的智能控制，根据用户行为和偏好自动调整播放内容和音量。

- **智能家居助手**：通过语音助手和控制器，实现对家中各种设备的智能控制，根据用户语音命令自动执行操作。

这些应用场景展示了MVC设计模式在家居自动化系统中的广泛适用性和强大功能。通过MVC设计模式，智能家居系统能够实现高度的模块化和可扩展性，为家庭生活带来更加便捷、舒适和智能的体验。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

智能家居系统中的数据模型通常使用Java中的基本数据类型或自定义类来实现。例如，可以定义一个`EnvironmentData`类，用于存储和处理家庭环境数据：

```java
public class EnvironmentData {
    private double temperature;
    private double humidity;
    private boolean smokeDetected;
    private boolean doorOpen;
    // getters and setters
}
```

### 4.2 公式推导过程

智能家居系统中的数据模型通常不需要进行复杂的数学计算，因此没有特定的公式推导过程。但是，在数据处理和分析过程中，可能会涉及一些基本的数学计算，例如平均值计算、最小值和最大值计算等。例如，假设要计算家庭环境数据的平均值，可以使用以下代码：

```java
public class EnvironmentData {
    private double temperature;
    private double humidity;
    private boolean smokeDetected;
    private boolean doorOpen;
    
    public double getAverage() {
        return (temperature + humidity) / 2;
    }
    
    // getters and setters
}
```

### 4.3 案例分析与讲解

以下是一个简单的智能家居系统案例，展示了如何使用MVC设计模式实现家居自动化功能。假设要实现对家中灯光的智能控制，可以使用以下代码：

```java
public class HomeApplication extends JFrame {
    private EnvironmentData environmentData;
    private HomeController homeController;
    
    public HomeApplication() {
        // 创建数据模型和控制器
        environmentData = new EnvironmentData();
        homeController = new HomeController(environmentData);
        
        // 创建界面元素并设置布局
        JLabel temperatureLabel = new JLabel("Temperature: ");
        JLabel humidityLabel = new JLabel("Humidity: ");
        JButton controlButton = new JButton("Control Devices");
        
        // 设置布局和显示界面
        setLayout(new GridLayout(3, 2));
        add(temperatureLabel);
        add(new JLabel(String.format("%.1f", environmentData.getTemperature())));
        add(humidityLabel);
        add(new JLabel(String.format("%.1f", environmentData.getHumidity())));
        add(controlButton);
        
        // 添加控制按钮事件处理
        controlButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                homeController.controlDevices("light", true);
            }
        });
        
        // 监听环境数据变化事件
        environmentData.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                if ("temperature".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                } else if ("humidity".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                } else if ("doorOpen".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                }
            }
        });
    }
    
    public static void main(String[] args) {
        new HomeApplication().setVisible(true);
    }
}
```

在这个案例中，`EnvironmentData`类负责存储家庭环境数据，`HomeController`类负责处理用户控制指令，`HomeApplication`类负责创建用户界面和事件处理。通过MVC设计模式，各个模块之间的交互透明化，便于维护和扩展。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

在进行智能家居系统开发前，我们需要准备好开发环境。以下是使用Java进行IDEA开发的环境配置流程：

1. 安装IntelliJ IDEA：从官网下载并安装IntelliJ IDEA，用于创建Java项目。

2. 创建并激活Maven项目：
```bash
mvn archetype:generate -DarchetypeArtifactId=maven-archetype-quickstart -DarchetypeVersion=1.0 -DgroupId=com.example -DartifactId=home-automation
mvn install:install-file -Dfile=project.xml -DgroupId=com.example -DartifactId=home-automation -Dpackaging=jar -Dversion=1.0
```

3. 安装Java库依赖：
```bash
mvn clean install
```

4. 安装Maven插件：
```bash
mvn install:install-file -Dfile=pom.xml -DgroupId=com.example -DartifactId=home-automation -Dversion=1.0
```

5. 编写代码：
```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

public class HomeApplication extends JFrame {
    private EnvironmentData environmentData;
    private HomeController homeController;
    
    public HomeApplication() {
        // 创建数据模型和控制器
        environmentData = new EnvironmentData();
        homeController = new HomeController(environmentData);
        
        // 创建界面元素并设置布局
        JLabel temperatureLabel = new JLabel("Temperature: ");
        JLabel humidityLabel = new JLabel("Humidity: ");
        JButton controlButton = new JButton("Control Devices");
        
        // 设置布局和显示界面
        setLayout(new GridLayout(3, 2));
        add(temperatureLabel);
        add(new JLabel(String.format("%.1f", environmentData.getTemperature())));
        add(humidityLabel);
        add(new JLabel(String.format("%.1f", environmentData.getHumidity())));
        add(controlButton);
        
        // 添加控制按钮事件处理
        controlButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                homeController.controlDevices("light", true);
            }
        });
        
        // 监听环境数据变化事件
        environmentData.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {
                if ("temperature".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                } else if ("humidity".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                } else if ("doorOpen".equals(evt.getPropertyName())) {
                    temperatureLabel.setText("Temperature: ");
                    JLabel temperatureLabel = new JLabel(String.format("%.1f", environmentData.getTemperature()));
                    JLabel humidityLabel = new JLabel(String.format("%.1f", environmentData.getHumidity()));
                    add(temperatureLabel);
                    add(humidityLabel);
                }
            }
        });
    }
    
    public static void main(String[] args) {
        new HomeApplication().setVisible(true);
    }
}
```

### 5.2 源代码详细实现

以下是使用Java Swing实现智能家居系统的完整代码：

```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

public class EnvironmentData {
    private double temperature;
    private double humidity;
    private boolean smokeDetected;
    private boolean doorOpen;
    
    public EnvironmentData() {
        temperature = 25.0;
        humidity = 50.0;
        smokeDetected = false;
        doorOpen = false;
    }
    
    public double getTemperature() {
        return temperature;
    }
    
    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }
    
    public double getHumidity() {
        return humidity;
    }
    
    public void setHumidity(double humidity) {
        this.humidity = humidity;
    }
    
    public boolean isSmokeDetected() {
        return smokeDetected;
    }
    
    public void setSmokeDetected(boolean smokeDetected) {
        this.smokeDetected = smokeDetected;
    }
    
    public boolean isDoorOpen() {
        return doorOpen;
    }
    
    public void setDoorOpen(boolean doorOpen) {
        this.doorOpen = doorOpen;
    }
    
    public double getAverage() {
        return (temperature + humidity) / 2;
    }
}

public class HomeController {
    private EnvironmentData environmentData;
    
    public HomeController(EnvironmentData environmentData) {
        this.environmentData = environmentData;
    }
    
    public void controlDevices(String command, boolean value) {
        if (command.equals("light")) {
            environmentData.setSmokeDetected(value);
        } else if (command.equals("heat")) {
            environmentData.setTemperature(environmentData.getTemperature() + 1);
        } else if (command.equals("cool")) {
            environmentData.setTemperature(environmentData.getTemperature() - 1);
        } else if (command.equals("vent")) {
            environmentData.setDoorOpen(!environmentData.isDoorOpen());
        }
    }
}

public class HomeApplication extends JFrame {
    private EnvironmentData environmentData;
    private HomeController homeController;
    
    public HomeApplication() {
        // 创建数据模型和控制器
        environmentData = new EnvironmentData();
        homeController = new HomeController(environmentData);
        
        // 创建界面元素并设置布局
        JLabel temperatureLabel = new JLabel("Temperature: ");
        JLabel humidityLabel = new JLabel("Humidity: ");
        JButton controlButton = new JButton("Control Devices");
        
        // 设置布局和显示界面
        setLayout(new GridLayout(3, 2));
        add(temperatureLabel);
        add(new JLabel(String.format("%.1f", environmentData.getTemperature())));
        add(humidityLabel);
        add(new JLabel(String.format("%.1f", environmentData.getHumidity())));
        add(controlButton);
        
        // 添加控制按钮事件处理
        controlButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                homeController.controlDevices("light", true);
            }
        });
        
        // 监听环境数据变化事件
        environmentData.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent evt) {



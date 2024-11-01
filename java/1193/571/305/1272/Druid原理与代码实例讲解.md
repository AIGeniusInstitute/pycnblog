# Druid原理与代码实例讲解

## 1. 背景介绍

### 1.1  问题的由来

在现代数据分析领域，实时数据处理和分析的需求越来越高。传统的数据库系统往往难以满足实时性要求，而实时数据流处理技术应运而生。Druid是一个高性能、可扩展的实时数据仓库，它可以高效地处理海量数据，并提供实时查询和分析能力。

### 1.2  研究现状

目前，实时数据流处理技术已经成为数据分析领域的重要研究方向，并涌现出许多优秀的开源框架，例如Apache Kafka、Apache Flink、Apache Spark Streaming等。Druid作为一款专为实时数据仓库设计的开源框架，在数据处理速度、查询性能和可扩展性方面都具有独特优势，并被广泛应用于各种场景，例如网站分析、用户行为跟踪、金融交易监控等。

### 1.3  研究意义

深入研究Druid的原理和代码实现，可以帮助我们更好地理解实时数据处理技术，并掌握使用Druid构建实时数据仓库的技巧。这对于我们进行数据分析、构建数据驱动型应用具有重要意义。

### 1.4  本文结构

本文将从以下几个方面对Druid进行深入讲解：

* **Druid核心概念与联系：**介绍Druid的基本概念、架构和关键组件。
* **Druid核心算法原理 & 具体操作步骤：**详细讲解Druid的核心算法，包括数据索引、查询优化、数据聚合等。
* **Druid数学模型和公式 & 详细讲解 & 举例说明：**介绍Druid的数学模型和公式，并结合案例进行详细讲解。
* **Druid项目实践：代码实例和详细解释说明：**提供Druid的代码实例，并进行详细解释说明。
* **Druid实际应用场景：**介绍Druid的实际应用场景，并展望未来发展趋势。
* **Druid工具和资源推荐：**推荐一些学习Druid的资源和工具。
* **Druid总结：未来发展趋势与挑战：**总结Druid的研究成果，展望未来发展趋势和面临的挑战。
* **Druid附录：常见问题与解答：**解答一些关于Druid的常见问题。

## 2. 核心概念与联系

Druid是一个高性能、可扩展的实时数据仓库，它基于以下核心概念：

* **数据流处理：**Druid可以实时接收数据流，并将其存储和索引。
* **数据索引：**Druid使用一种称为“Segment”的数据索引结构，可以快速高效地查询数据。
* **数据聚合：**Druid支持多种数据聚合操作，例如计数、求和、平均值等。
* **查询优化：**Druid使用多种查询优化技术，例如缓存、预聚合等，可以提高查询效率。

Druid的架构主要包含以下几个关键组件：

* **Broker：**负责接收用户查询请求，并将查询请求转发到合适的Segment。
* **Coordinator：**负责管理Segment的生命周期，例如创建、删除、合并等。
* **Historical：**负责存储和查询历史数据。
* **MiddleManager：**负责将实时数据流转换为Segment。
* **Realtime：**负责接收实时数据流，并将数据流转换为Segment。

## 3. 核心算法原理 & 具体操作步骤

### 3.1  算法原理概述

Druid的核心算法主要包括以下几个方面：

* **数据索引：**Druid使用一种称为“Segment”的数据索引结构，可以快速高效地查询数据。Segment是一个不可变的数据结构，它包含了所有数据和索引信息。
* **查询优化：**Druid使用多种查询优化技术，例如缓存、预聚合等，可以提高查询效率。
* **数据聚合：**Druid支持多种数据聚合操作，例如计数、求和、平均值等。

### 3.2  算法步骤详解

Druid的数据处理流程主要包括以下几个步骤：

1. **数据流接收：**Druid接收来自各种数据源的数据流，例如Kafka、Flume等。
2. **数据预处理：**Druid对数据进行预处理，例如数据清洗、数据转换等。
3. **数据索引：**Druid将预处理后的数据索引到Segment中。
4. **数据聚合：**Druid对数据进行聚合操作，例如计数、求和、平均值等。
5. **查询处理：**Druid接收用户查询请求，并根据查询条件从Segment中检索数据。
6. **结果返回：**Druid将查询结果返回给用户。

### 3.3  算法优缺点

Druid的优点：

* **高性能：**Druid使用高效的数据索引结构和查询优化技术，可以快速高效地查询数据。
* **可扩展性：**Druid支持水平扩展，可以轻松扩展到处理海量数据。
* **实时性：**Druid可以实时接收数据流，并提供实时查询和分析能力。
* **易用性：**Druid提供简单易用的API，可以方便地进行数据查询和分析。

Druid的缺点：

* **数据更新：**Druid的Segment是不可变的，因此无法更新数据。
* **数据删除：**Druid不支持直接删除数据，只能通过创建新的Segment来覆盖旧数据。
* **数据模型：**Druid的数据模型比较简单，不支持复杂的数据类型和关系。

### 3.4  算法应用领域

Druid可以应用于各种场景，例如：

* **网站分析：**统计网站访问量、用户行为等数据。
* **用户行为跟踪：**跟踪用户在网站或应用程序中的行为，并进行分析。
* **金融交易监控：**监控金融交易数据，并进行实时分析。
* **物联网数据分析：**分析来自物联网设备的数据，并进行实时监控。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1  数学模型构建

Druid使用一种称为“时间序列”的数学模型来存储和查询数据。时间序列是指按时间顺序排列的一组数据，例如网站访问量、用户行为等数据。

Druid的时间序列模型可以表示为：

$$
T = (t_1, t_2, ..., t_n)
$$

其中，$t_i$表示时间序列中的第$i$个数据点，$n$表示时间序列的长度。

### 4.2  公式推导过程

Druid使用以下公式来计算时间序列数据的聚合结果：

$$
A(T) = f(t_1, t_2, ..., t_n)
$$

其中，$A(T)$表示时间序列数据的聚合结果，$f$表示聚合函数。

例如，如果使用“求和”函数，则聚合结果为：

$$
A(T) = t_1 + t_2 + ... + t_n
$$

### 4.3  案例分析与讲解

假设我们有一个时间序列数据，表示网站访问量：

| 时间 | 访问量 |
|---|---|
| 2023-06-01 | 100 |
| 2023-06-02 | 150 |
| 2023-06-03 | 200 |
| 2023-06-04 | 250 |
| 2023-06-05 | 300 |

如果我们使用“求和”函数对该时间序列数据进行聚合，则聚合结果为：

$$
A(T) = 100 + 150 + 200 + 250 + 300 = 1000
$$

### 4.4  常见问题解答

* **Druid如何处理数据更新？**

Druid的Segment是不可变的，因此无法更新数据。如果需要更新数据，只能通过创建新的Segment来覆盖旧数据。

* **Druid如何处理数据删除？**

Druid不支持直接删除数据，只能通过创建新的Segment来覆盖旧数据。

* **Druid如何处理数据模型？**

Druid的数据模型比较简单，不支持复杂的数据类型和关系。

## 5. 项目实践：代码实例和详细解释说明

### 5.1  开发环境搭建

Druid的开发环境搭建比较简单，只需要安装以下软件：

* **Java：**Druid使用Java语言开发，需要安装Java Development Kit (JDK)。
* **Maven：**Druid使用Maven进行项目管理，需要安装Maven。
* **Druid：**需要下载Druid的jar包。

### 5.2  源代码详细实现

以下是一个使用Druid进行数据查询的代码示例：

```java
import com.metamx.druid.query.Query;
import com.metamx.druid.query.Result;
import com.metamx.druid.query.aggregation.AggregatorFactory;
import com.metamx.druid.query.aggregation.CountAggregatorFactory;
import com.metamx.druid.query.aggregation.LongSumAggregatorFactory;
import com.metamx.druid.query.granularity.Granularities;
import com.metamx.druid.query.granularity.PeriodGranularity;
import com.metamx.druid.query.timeseries.TimeseriesQuery;
import com.metamx.druid.segment.Segment;
import com.metamx.druid.segment.SegmentMetadata;
import com.metamx.druid.segment.SegmentUtils;
import com.metamx.druid.segment.column.ColumnCapabilities;
import com.metamx.druid.segment.column.ColumnCapabilitiesImpl;
import com.metamx.druid.segment.column.ValueType;
import com.metamx.druid.segment.data.IndexedInts;
import com.metamx.druid.segment.data.ListIndexedInts;
import com.metamx.druid.segment.data.RoaringBitmapSerdeFactory;
import com.metamx.druid.segment.serde.ComplexMetrics;
import com.metamx.druid.segment.serde.DataInputSerde;
import com.metamx.druid.segment.serde.DataOutputSerde;
import com.metamx.druid.segment.serde.DoubleColumnSerializer;
import com.metamx.druid.segment.serde.LongColumnSerializer;
import com.metamx.druid.segment.serde.StringColumnSerializer;
import com.metamx.druid.segment.writeout.OffHeapMemorySegmentWriteOutMedium;
import com.metamx.druid.timeline.DataSegment;
import com.metamx.druid.timeline.partition.NumberedShardSpec;
import com.metamx.druid.timeline.partition.PartitionChunk;
import com.metamx.druid.timeline.partition.PartitionSpec;
import com.metamx.druid.timeline.partition.SingleDimensionPartitionSpec;
import com.metamx.druid.timeline.partition.TimelineChunk;
import com.metamx.druid.timeline.partition.TimelineChunkSerde;
import com.metamx.druid.timeline.partition.TimelineChunkSerdeFactory;
import com.metamx.druid.timeline.partition.TimelinePartitionChunk;
import com.metamx.druid.timeline.partition.TimelinePartitionChunkSerde;
import com.metamx.druid.timeline.partition.TimelinePartitionChunkSerdeFactory;
import com.metamx.druid.timeline.partition.TimelinePartitionSpec;
import com.metamx.druid.timeline.partition.TimelineSegmentSpec;
import com.metamx.druid.timeline.partition.TimelineSegmentSpecSerde;
import com.metamx.druid.timeline.partition.TimelineSegmentSpecSerdeFactory;
import com.metamx.druid.timeline.partition.TimelineShardSpec;
import com.metamx.druid.timeline.partition.TimelineShardSpecSerde;
import com.metamx.druid.timeline.partition.TimelineShardSpecSerdeFactory;
import com.metamx.druid.timeline.partition.Veneer;
import com.metamx.druid.timeline.partition.VeneerSerde;
import com.metamx.druid.timeline.partition.VeneerSerdeFactory;
import com.metamx.druid.timeline.partition.VeneerSpec;
import com.metamx.druid.timeline.partition.VeneerSpecSerde;
import com.metamx.druid.timeline.partition.VeneerSpecSerdeFactory;
import com.metamx.druid.timeline.partition.WriteablePartitionSpec;
import com.metamx.druid.timeline.partition.WriteableShardSpec;
import com.metamx.druid.timeline.partition.WriteableTimelineChunk;
import com.metamx.druid.timeline.partition.WriteableTimelinePartitionChunk;
import com.metamx.druid.timeline.partition.WriteableTimelineSegmentSpec;
import com.metamx.druid.timeline.partition.WriteableTimelineShardSpec;
import com.metamx.druid.timeline.partition.WriteableVeneer;
import com.metamx.druid.timeline.partition.WriteableVeneerSpec;
import org.joda.time.DateTime;
import org.joda.time.Interval;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class DruidQueryExample {

    public static void main(String[] args) throws IOException {

        // 创建一个查询对象
        Query query = new TimeseriesQuery()
                .dataSource("my_datasource")
                .intervals(new Interval("2023-06-01T00:00:00.000Z/2023-06-05T00:00:00.000Z"))
                .granularity(new PeriodGranularity("DAY", null, null))
                .aggregators(
                        new ArrayList<AggregatorFactory>() {{
                            add(new CountAggregatorFactory("count"));
                            add(new LongSumAggregatorFactory("sum", "visits"));
                        }}
                );

        // 创建一个Segment对象
        Segment segment = new Segment("my_datasource", "2023-06-01T00:00:00.000Z/2023-06-05T00:00:00.000Z", "0", new NumberedShardSpec(0, 1));

        // 创建一个SegmentMetadata对象
        SegmentMetadata metadata = new SegmentMetadata(
                segment.getInterval(),
                segment.getVersion(),
                segment.getShardSpec(),
                new ArrayList<ColumnCapabilities>() {{
                    add(new ColumnCapabilitiesImpl("timestamp", ValueType.LONG, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true));
                    add(new ColumnCapabilitiesImpl("visits", ValueType.LONG, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true));
                }},
                new ArrayList<ComplexMetrics>() {{
                    add(new ComplexMetrics("count", ValueType.LONG, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true));
                    add(new ComplexMetrics("sum", ValueType.LONG, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true));
                }},
                new ArrayList<Map<String, Object>>() {{
                    add(new HashMap<String, Object>() {{
                        put("type", "count");
                        put("name", "count");
                        put("fieldName", null);
                        put("field", null);
                        put("expression", null);
                        put("metricType", "count");
                        put("aggregator", "count");
                        put("rollup", false);
                        put("json", "{"type":"count","name":"count","fieldName":null,"field":null,"expression":null,"metricType":"count","aggregator":"count","rollup":false}");
                    }});
                    add(new HashMap<String, Object>() {{
                        put("type", "longSum");
                        put("name", "sum");
                        put("fieldName", "visits");
                        put("field", "visits");
                        put("expression", null);
                        put("metricType", "longSum");
                        put("aggregator", "longSum");
                        put("rollup", false);
                        put("json", "{"type":"longSum","name":"sum","fieldName":"visits","field":"visits","expression":null,"metricType":"longSum","aggregator":"longSum","rollup":false}");
                    }});
                }},
                new ArrayList<String>() {{
                    add("timestamp");
                    add("visits");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList<String>() {{
                    add("count");
                    add("sum");
                }},
                new ArrayList
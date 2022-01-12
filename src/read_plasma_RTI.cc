#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <arrow/api.h>
#include <plasma/client.h>
#include <plasma/common.h>
#include <arrow/util/logging.h>
#include <arrow/tensor.h>
#include <arrow/array.h>
#include <arrow/table.h>
#include <arrow/buffer.h>
#include <arrow/type_fwd.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/datum.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/compute/api_vector.h>

#include <boost/date_time/date.hpp>
#include <boost/date_time.hpp>

typedef std::vector<std::shared_ptr<arrow::Table>> TableVector;

int64_t to_unixtime(boost::posix_time::ptime t) {
    auto diff = t - boost::posix_time::ptime(boost::gregorian::date(1970, boost::gregorian::Jan, 1));
    return diff.total_seconds();
}

TableVector read_tables(plasma::PlasmaClient& client) {
    plasma::ObjectTable objTable;
    ARROW_CHECK_OK(client.List(&objTable));
    std::vector<plasma::ObjectID> obj_ids(objTable.size());
    std::size_t i{0};
    for (auto const& [obj_id, entry] : objTable)
        obj_ids[i++] = obj_id;
    std::vector<plasma::ObjectBuffer> buffers;
    ARROW_CHECK_OK(client.Get(obj_ids, -1, &buffers));
    std::vector<arrow::io::BufferReader> buffer_readers;
    TableVector tables;
    for(const auto& buffer : buffers) {
        auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer.data);
        auto result = arrow::ipc::RecordBatchStreamReader::Open(buffer_reader);
        if(result.ok()) {
            auto stream_reader = result.ValueOrDie();
            auto result = arrow::Table::FromRecordBatchReader(&*stream_reader);
            if(result.ok()) {
                auto table = result.ValueOrDie();
                tables.emplace_back(table);
            }
            else {
                ARROW_LOG(ERROR) << result.status();
            }
        }
        else {
            ARROW_LOG(ERROR) << result.status();
        }
    }
    return tables;
}

void read_vectors(const std::shared_ptr<arrow::Table>& table) {
    uint8_t time_resolution{1}, height_resolution{5};
    int min_height{200}, max_height{800};
    auto datetime_col = table->GetColumnByName(std::string("datetime"));
    auto heights_col = table->GetColumnByName(std::string("GDALT"));
    const auto l = datetime_col->length();
    int64_t year;
    {
        auto _date = datetime_col->GetScalar(l/2).ValueOrDie();
        auto _year = *static_cast<int64_t *>(
                        const_cast<arrow::Int64Scalar&>(
                            arrow::compute::Year(arrow::Datum(_date)).ValueOrDie().scalar_as<arrow::Int64Scalar>())
                                    .mutable_data());
        year = _year;
    }
    {
        namespace bg = boost::gregorian;
        namespace bpt = boost::posix_time;
        namespace ac = arrow::compute;
        bg::day_iterator day_itr(bg::date(year, bg::Jan, 1));
        auto _end_date = bg::date(year, bg::Dec, 31);
        size_t i{0};
        do {
            // std::cout << to_simple_string(*day_itr) << std::endl;
            auto day = day_itr->day();
            auto month = day_itr->month();
            
            // auto second_unit = ;
            // second_unit()
            auto SECONDS = arrow::timestamp(arrow::TimeUnit::SECOND);
            arrow::TimestampScalar x(0, SECONDS);
            arrow::TimestampScalar y(1, SECONDS);
            // std::cout << x.ToString() << " " << y.ToString() << std::endl;
            auto _some_date = bg::date(2020, 10, 30);
            arrow::TimestampScalar z(to_unixtime(bpt::ptime(_some_date)), SECONDS);
            // std::cout << "special date " << z.ToString() << std::endl;
            // std::cout << "x value: " << x.value << std::endl;
            auto start_t = bpt::ptime(bg::date(year, month, day), bpt::time_duration(19, 0, 0));
            auto end_t = bpt::ptime(bg::date(year, month, day) + bg::days(1), bpt::time_duration(7, 0, 0));
            
            auto start_t_unix = to_unixtime(start_t);
            auto end_t_unix = to_unixtime(end_t);

            std::vector<bool> bitmap(l, false);
            int count{0};
            for(; i < l; ++i) {
                auto height = std::static_pointer_cast<arrow::DoubleScalar>(heights_col->GetScalar(i).ValueOrDie())->value;
                if(height < min_height || height > max_height) continue;
                int64_t t_unix = std::static_pointer_cast<arrow::TimestampScalar>(datetime_col->GetScalar(i).ValueOrDie())->value / 1e9;
                // std::cout << "comparing t_unix " << t_unix << " with end_t_unix " << end_t_unix << std::endl;
                // std::cout << "comparing t_unix " << t_unix << " with start_t_unix " << start_t_unix << std::endl;
                bool valid = true;
                while(t_unix >= end_t_unix && ++day_itr <= _end_date) {
                    valid = false;
                    day = day_itr->day();
                    month = day_itr->month();

                    start_t = bpt::ptime(bg::date(year, month, day), bpt::time_duration(19, 0, 0));
                    end_t = bpt::ptime(bg::date(year, month, day) + bg::days(1), bpt::time_duration(7, 0, 0));
                    
                    start_t_unix = to_unixtime(start_t);
                    end_t_unix = to_unixtime(end_t);
                    if(t_unix < end_t_unix) {
                        --day_itr;
                        break;
                    }
                }
                if(!valid) break;
                if(t_unix >= start_t_unix) bitmap[i] = true, ++count;
            }
            if(count) {
                std::cout << "number of observations between " << bpt::to_simple_string(start_t) << " and " << bpt::to_simple_string(end_t) << " is: " << count << std::endl;
                arrow::BooleanBuilder builder;
                ARROW_CHECK_OK(builder.Reserve(l));
                ARROW_CHECK_OK(builder.AppendValues(bitmap));
                auto array = builder.Finish().ValueOrDie();
                auto filtered = ac::Filter(arrow::Datum(table), arrow::Datum(array)).ValueOrDie().table();
                auto filtered_datetime = filtered->GetColumnByName(std::string("datetime"));
                auto x = filtered_datetime->GetScalar(0).ValueOrDie();
                std::cout << x->ToString() << std::endl;
            }
        } while(++day_itr <= _end_date);
    }
}

int main() {
    plasma::PlasmaClient client;
    ARROW_CHECK_OK(client.Connect("/tmp/plasma", "", 0));
    auto tables = read_tables(client);
    for(const auto& table: tables)
        for(const auto& field : table->fields())
            std::cout << field->ToString() << std::endl;
    for(const auto& table : tables)
        read_vectors(table);

    ARROW_CHECK_OK(client.Disconnect());
}


















// #include <arrow/api.h>
// #include <plasma/client.h>
// #include <arrow/util/logging.h>
// #include <arrow/tensor.h>
// #include <arrow/array.h>
// #include <arrow/buffer.h>
// #include <arrow/io/memory.h>
// #include <arrow/ipc/writer.h>
// using namespace plasma;
// using namespace arrow;

// int main(int argc, char** argv) {
//   // Start up and connect a Plasma client.
//   PlasmaClient client;
//   ARROW_CHECK_OK(client.Connect("/tmp/plasma", "", 0));
//   // Generate an Object ID for Plasma
//   ObjectID object_id = ObjectID::from_binary("11111111111111111111");

//   // Generate Float Array
//   int64_t input_length = 1000;
//   std::vector<float> input(input_length);
//   for (int64_t i = 0; i < input_length; ++i) {
//     input[i] = 2.0;
//   }

//   // Create Arrow Tensor Object, no copy made!
//   // {input_length} is the shape of the tensor
//   auto value_buffer = Buffer::Wrap<float>(input);
//   Tensor t(float32(), value_buffer, {input_length});

//   // Get the size of the tensor to be stored in Plasma
//   int64_t datasize;
//   ARROW_CHECK_OK(ipc::GetTensorSize(t, &datasize));
//   int32_t meta_len = 0;

//   // Create the Plasma Object
//   // Plasma is responsible for initializing and resizing the buffer
//   // This buffer will contain the _serialized_ tensor
//   std::shared_ptr<Buffer> buffer;
//   ARROW_CHECK_OK(
//       client.Create(object_id, datasize, NULL, 0, &buffer));

//   // Writing Process, this will copy the tensor into Plasma
//   io::FixedSizeBufferWriter stream(buffer);
//   ARROW_CHECK_OK(arrow::ipc::WriteTensor(t, &stream, &meta_len, &datasize));

//   // Seal Plasma Object
//   // This computes a hash of the object data by default
//   ARROW_CHECK_OK(client.Seal(object_id));
// }